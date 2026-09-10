"""
P5: 网关前置语义缓存 (Milvus)

问题 → 嵌入 → Milvus 语义相似度检索:
  L1 (score ≥ 0.93) 近精确命中 → 直接返回缓存答案 (毫秒级)
  L2 (score ≥ 0.82) 语义相似   → 返回缓存答案并标记 semantic_hit (带防误命守卫)
  miss → 走完整管道, 完成后 set_cache 回写

守卫 (防历史"同公司不同年份串答"类误命中):
  - 年份 (\\d{4}) 不一致 → 拒绝
  - 订单号 (SO\\d+) 不一致 → 拒绝
  - n-gram 词面重叠过低 → 拒绝 (向量像但问题不同)

依赖: Milvus 不可用时优雅降级为 miss (不阻塞主流程, 回落现有 SQLite 缓存)。
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from src.cache.privacy import is_private_or_money_query

CACHE_COLLECTION = "kefu_semantic_cache"
L1_THRESHOLD = float(os.getenv("CACHE_L1_THRESHOLD", "0.93"))
L2_THRESHOLD = float(os.getenv("CACHE_L2_THRESHOLD", "0.82"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "2000"))
TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "1"))
DIM = int(os.getenv("MILVUS_DIM", "768"))

_YEAR_RE = re.compile(r"20\d{2}")
_ORDER_RE = re.compile(r"SO\d+")


class GatewayCache:
    _instance: Optional['GatewayCache'] = None

    def __new__(cls) -> 'GatewayCache':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._ok = False
        self._mem: dict = {}  # 降级: 进程内 {question: result}

    def _ensure(self) -> bool:
        if self._ok:
            return True
        try:
            from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
            connections.connect(alias="default", host=os.getenv("MILVUS_HOST", "127.0.0.1"),
                                port=os.getenv("MILVUS_PORT", "19530"))
            if not utility.has_collection(CACHE_COLLECTION):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
                    FieldSchema(name="answer_json", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="years", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="order_ids", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="ts", dtype=DataType.INT64),
                ]
                col = Collection(CACHE_COLLECTION, CollectionSchema(fields, "问答语义缓存"))
                col.create_index("embedding", {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}})
            Collection(CACHE_COLLECTION).load()
            self._ok = True
            logger.info("GatewayCache: Milvus 语义缓存就绪")
        except Exception as e:
            self._ok = False
            logger.warning("GatewayCache: Milvus 不可用, 语义缓存降级内存: %s", e)
        return self._ok

    def _embed(self, text: str):
        from src.retrieval.retriever import HybridRetriever
        return HybridRetriever().model.encode([text], normalize_embeddings=True)[0]

    # ── 守卫 ──────────────────────────────────────

    @staticmethod
    def _sig(q: str) -> tuple:
        return (tuple(sorted(set(_YEAR_RE.findall(q)))), tuple(sorted(set(_ORDER_RE.findall(q)))))

    @staticmethod
    def _overlap(q1: str, q2: str) -> bool:
        """词面重叠检查: 2-gram 命中率 ≥ 0.4 才算同题 (向量像但词面完全不同 → 拒绝)"""
        g1, g2 = {q1[i:i + 2] for i in range(len(q1) - 1)}, {q2[i:i + 2] for i in range(len(q2) - 1)}
        if not g1 or not g2:
            return True
        return len(g1 & g2) / min(len(g1), len(g2)) >= 0.4

    def _guard_pass(self, cached_q: str, query: str) -> bool:
        # 年份/订单号不一致 → 拒绝 (历史误命中的根源)
        if self._sig(cached_q) != self._sig(query):
            return False
        return self._overlap(cached_q, query)

    # ── 读 ────────────────────────────────────────

    def get_cached(self, query: str) -> Optional[dict]:
        """网关前置查询. 命中返回缓存答案 (L2 命中带 semantic_hit=True), 否则 None"""
        if not query.strip():
            return None
        if is_private_or_money_query(query):
            return None  # 用户私有/动账类不读共享缓存
        if not self._ensure():
            return self._mem.get(query)
        try:
            from pymilvus import Collection
            emb = self._embed(query)
            col = Collection(CACHE_COLLECTION)
            res = col.search(
                data=[emb.tolist()], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 8}},
                limit=1, output_fields=["question", "answer_json", "ts"],
            )
            if not res[0]:
                return None
            hit = res[0][0]
            score = float(hit.score)
            if score < L2_THRESHOLD:
                return None
            cached_q = hit.entity.get("question") or ""
            # 守卫对 L1 同样生效: 高相似度但年份/订单号不同 → 不同问题, 拒绝
            sig_ok = self._sig(cached_q) == self._sig(query)
            if score >= L1_THRESHOLD:
                if not sig_ok:
                    logger.debug("GatewayCache L1 守卫拒绝: %s vs %s", cached_q, query)
                    return None
                try:
                    result = json.loads(hit.entity.get("answer_json") or "{}")
                except Exception:
                    return None
                result["semantic_hit"] = True
                logger.info("GatewayCache L1 命中: score=%.3f %s", score, query[:30])
                return result
            if score >= L2_THRESHOLD and sig_ok and self._overlap(cached_q, query):
                try:
                    result = json.loads(hit.entity.get("answer_json") or "{}")
                except Exception:
                    return None
                result["semantic_hit"] = False
                logger.info("GatewayCache L2 命中: score=%.3f %s", score, query[:30])
                return result
            logger.debug("GatewayCache 守卫拒绝: score=%.3f 但问题不同", score)
            return None
        except Exception as e:
            logger.debug("GatewayCache 查询失败: %s", e)
            return None

    # ── 写 ────────────────────────────────────────

    def set_cache(self, query: str, result: dict) -> None:
        """回写缓存 (幂等)"""
        if not query.strip() or not result.get("answer"):
            return
        if is_private_or_money_query(query):
            return  # 用户私有/动账类不写共享缓存
        if not self._ensure():
            self._mem[query] = result
            if len(self._mem) > MAX_ENTRIES:
                self._mem.clear()
            return
        try:
            from pymilvus import Collection
            years = ",".join(sorted(set(_YEAR_RE.findall(query))))
            order_ids = ",".join(sorted(set(_ORDER_RE.findall(query))))
            col = Collection(CACHE_COLLECTION)
            # 幂等: 先删同 question 旧记录
            col.delete(expr=f'question == "{query[:200]}"')
            col.insert([{
                "question": query[:500], "embedding": self._embed(query).tolist(),
                "answer_json": json.dumps(result, ensure_ascii=False)[:60000],
                "years": years, "order_ids": order_ids, "ts": int(time.time()),
            }])
            col.flush()
        except Exception as e:
            logger.debug("GatewayCache 写入失败: %s", e)

    def clear(self) -> None:
        if self._ensure():
            try:
                from pymilvus import Collection
                Collection(CACHE_COLLECTION).drop()
                self._ok = False
            except Exception as e:
                logger.warning("GatewayCache 清空失败: %s", e)


# 全局单例
gateway_cache = GatewayCache()
