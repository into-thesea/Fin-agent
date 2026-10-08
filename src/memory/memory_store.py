"""
P4: 分层记忆 (短期会话滑动窗口 + 长期用户画像 + 向量语义记忆)

- 短期: Redis list `mem:session:{sid}`，每会话最近 WINDOW_SIZE 轮 (query, answer)
- 长期: Redis hash `mem:user:{uid}` 结构化画像 (产品兴趣等)
- 向量: Milvus 集合 `kefu_user_memory`，用户历史问题向量化，语义召回做个性化
- 融合: build_memory_context() 在请求入口合成文本注入，供路由/领域 Agent 使用

依赖降级: Redis/Milvus 不可用时静默降级为空记忆 (不阻塞主流程)。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))
MEM_SESSION_PREFIX = "mem:session:"
MEM_USER_PREFIX = "mem:user:"
MEM_USER_COLLECTION = "kefu_user_memory"
MEM_DIM = int(os.getenv("MILVUS_DIM", "768"))
# Collection.load() 不带 timeout 会无限轮询 (Milvus 半死状态下会挂住调用方)
CONNECT_TIMEOUT = float(os.getenv("MILVUS_CONNECT_TIMEOUT", "5"))
LOAD_TIMEOUT = float(os.getenv("MILVUS_LOAD_TIMEOUT", "10"))


class MemoryStore:
    _instance: Optional['MemoryStore'] = None

    def __new__(cls) -> 'MemoryStore':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._redis = None
        self._milvus_ok = False
        try:
            from src.cache.redis_client import RedisCache
            cache = RedisCache()
            if cache.ping():
                self._redis = cache.client
                logger.info("MemoryStore: Redis 已连接")
        except Exception as e:
            logger.debug("MemoryStore Redis 不可用 (降级): %s", e)
        if self._redis is None:
            logger.info("MemoryStore: Redis 不可用, 记忆降级为空 (不阻塞)")

    # ── Milvus 向量记忆 (惰性) ──────────────────────

    def _ensure_milvus(self) -> bool:
        """确保 kefu_user_memory 集合存在 (惰性, 失败降级)"""
        if self._milvus_ok:
            return True
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
            connections.connect(alias="default", host=os.getenv("MILVUS_HOST", "127.0.0.1"),
                                port=os.getenv("MILVUS_PORT", "19530"),
                                timeout=CONNECT_TIMEOUT)
            if not utility.has_collection(MEM_USER_COLLECTION, timeout=CONNECT_TIMEOUT):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MEM_DIM),
                ]
                col = Collection(MEM_USER_COLLECTION, CollectionSchema(fields, "用户长期记忆"))
                col.create_index("embedding", {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}})
            Collection(MEM_USER_COLLECTION).load(timeout=LOAD_TIMEOUT)
            self._milvus_ok = True
        except Exception as e:
            logger.warning("MemoryStore Milvus 不可用 (向量记忆降级): %s", e)
        return self._milvus_ok

    def _embed(self, text: str):
        """bge 向量化 (复用 HybridRetriever 已加载模型)"""
        from src.retrieval.retriever import HybridRetriever
        return HybridRetriever().model.encode([text], normalize_embeddings=True)[0]

    # ── 短期: 会话滑动窗口 ──────────────────────────

    def save_turn(self, session_id: str, user_id: str, query: str, answer: str,
                  entities: Optional[list] = None) -> None:
        """保存一轮对话到会话窗口 + 更新用户画像 + 写向量记忆"""
        if not self._redis:
            return
        try:
            key = MEM_SESSION_PREFIX + (session_id or "anon")
            item = json.dumps({"q": (query or "")[:300], "a": (answer or "")[:500],
                               "ts": time.time()}, ensure_ascii=False)
            self._redis.rpush(key, item)
            self._redis.ltrim(key, -WINDOW_SIZE, -1)
            self._redis.expire(key, 86400 * 7)
        except Exception as e:
            logger.debug("会话窗口写入失败 (可忽略): %s", e)
        if user_id:
            self._update_profile(user_id, query, entities)
            self._save_vector_memory(user_id, query)

    def get_session_window(self, session_id: str, n: int = WINDOW_SIZE) -> list:
        """最近 n 轮对话 [(user_q, bot_a), ...]"""
        if not self._redis:
            return []
        try:
            items = self._redis.lrange(MEM_SESSION_PREFIX + (session_id or "anon"), -n, -1)
            turns = []
            for raw in items:
                try:
                    d = json.loads(raw)
                    turns.append((d.get("q", ""), d.get("a", "")))
                except Exception:
                    continue
            return turns
        except Exception as e:
            logger.debug("会话窗口读取失败: %s", e)
            return []

    # ── 长期: 用户画像 ──────────────────────────────

    def _update_profile(self, user_id: str, query: str, entities: Optional[list]) -> None:
        """从查询/实体中抽取用户兴趣 (ponytail: 只记 Product 类型实体)"""
        if not self._redis:
            return
        interests = []
        for ent in entities or []:
            if ent.get("type") == "Product" and ent.get("name"):
                interests.append(ent["name"])
        if not interests:
            return
        try:
            key = MEM_USER_PREFIX + user_id
            existing = self._redis.hget(key, "interests") or ""
            merged = list(dict.fromkeys((existing.split(",") if existing else []) + interests))
            self._redis.hset(key, mapping={"interests": ",".join(merged[-10:]), "updated_at": int(time.time())})
        except Exception as e:
            logger.debug("用户画像更新失败: %s", e)

    def get_user_profile(self, user_id: str) -> dict:
        if not self._redis:
            return {}
        try:
            return self._redis.hgetall(MEM_USER_PREFIX + user_id) or {}
        except Exception as e:
            logger.debug("用户画像读取失败: %s", e)
            return {}

    # ── 长期: 向量语义记忆 (Milvus) ──────────────────

    def _save_vector_memory(self, user_id: str, query: str) -> None:
        if not self._ensure_milvus() or not query.strip():
            return
        try:
            from pymilvus import Collection
            emb = self._embed(query)
            col = Collection(MEM_USER_COLLECTION)
            col.insert([{
                "user_id": user_id, "content": query[:500], "embedding": emb.tolist(),
            }])
            col.flush()  # 立即落盘, 保证后续 search 可见
        except Exception as e:
            logger.debug("向量记忆写入失败 (可忽略): %s", e)

    def recall_user_memory(self, user_id: str, query: str, top_k: int = 2) -> list:
        """语义召回该用户的历史相关问题 (个性化参考)"""
        if not user_id or not query.strip() or not self._ensure_milvus():
            return []
        try:
            from pymilvus import Collection
            col = Collection(MEM_USER_COLLECTION)
            emb = self._embed(query)
            res = col.search(
                data=[emb.tolist()], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 4}},
                limit=top_k, expr=f'user_id == "{user_id}"',
                output_fields=["content"],
            )
            return [h.entity.get("content") or "" for h in res[0]]
        except Exception as e:
            logger.debug("向量记忆召回失败 (可忽略): %s", e)
            return []

    # ── 融合注入 ────────────────────────────────────

    def build_memory_context(self, session_id: str, user_id: str, query: str) -> str:
        """合成记忆上下文文本, 供注入 prompt"""
        parts = []
        turns = self.get_session_window(session_id)
        if turns:
            lines = [f"用户: {q} / 客服: {a[:80]}" for q, a in turns if q]
            parts.append("【最近对话】\n" + "\n".join(lines))
        profile = self.get_user_profile(user_id)
        if profile.get("interests"):
            parts.append(f"【该用户偏好】喜欢的产品: {profile['interests']}")
        if user_id:
            recalled = self.recall_user_memory(user_id, query)
            if recalled:
                parts.append("【该用户曾问过】" + " / ".join(f"「{r[:40]}」" for r in recalled))
        return "\n\n".join(parts)


# 全局单例
memory_store = MemoryStore()
