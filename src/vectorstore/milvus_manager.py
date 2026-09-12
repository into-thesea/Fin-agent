"""
Milvus 向量索引管理器 (P0)

接口镜像 FaissIndexManager，供 HybridRetriever 无缝切换：
  - Milvus 可用 → 用 Milvus（检索外置、可分布式、支撑记忆/语义缓存）
  - Milvus 不可用 → 自动降级 FAISS（现有路径不变，系统不崩）

设计要点:
  - pymilvus 懒加载：未安装或 Milvus 未启动时，模块 import 不报错，
    search/add 等方法静默返回空 / no-op，日志提示降级。
  - 向量距离：COSINE（与 FAISS IndexFlatIP + 归一化向量等价）。
  - 删除：Milvus 原生硬删除（小库场景最简单）。
"""

from __future__ import annotations

import os
import warnings
import threading
import logging
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# ponytail: ORM API (connections/Collection) 在 pymilvus 3.1 移除，届时迁移 MilvusClient；
# 当前 3.0.x 仍可用，静默弃用告警避免日志噪音（按消息匹配，不依赖告警类名）
warnings.filterwarnings("ignore", message=".*ORM-style PyMilvus API.*")

# 模块级懒导入：pymilvus 未安装时各方法经 available 短路，不会解引用 None
try:
    from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
except Exception:  # pragma: no cover - 未安装 pymilvus 时降级
    connections = utility = Collection = CollectionSchema = FieldSchema = DataType = None

# ── 配置 (可通过 .env 覆盖) ──
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "kefu_chunks")
MILVUS_DIM = int(os.getenv("MILVUS_DIM", "768"))


class MilvusManager:
    """Milvus 向量索引管理器 (全局单例，接口对齐 FaissIndexManager)"""

    _instance: Optional['MilvusManager'] = None
    _lock = threading.RLock()

    def __new__(cls) -> 'MilvusManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        with self._lock:
            if self._initialized:
                return
            self._collection_name = MILVUS_COLLECTION
            self._dim = MILVUS_DIM
            self._connected = False
            self._collection: Any = None
            self._check_done = False
            self._initialized = True
            logger.info(
                "MilvusManager 已初始化: %s:%s collection=%s",
                MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION,
            )

    # ──────────────────────────────────────────────
    # 连接管理（惰性，失败优雅降级）
    # ──────────────────────────────────────────────

    def _ensure_ready(self) -> bool:
        """首次使用时尝试连接 + 建 collection。失败则标记不可用，不抛异常。"""
        if self._check_done:
            return self._connected
        with self._lock:
            if self._check_done:
                return self._connected
            self._check_done = True
            try:
                connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
                self._connected = True
                logger.info("Milvus 连接成功: %s:%s", MILVUS_HOST, MILVUS_PORT)
            except Exception as e:
                self._connected = False
                logger.warning("Milvus 不可用 (%s), 向量检索降级到 FAISS", e)
            if self._connected:
                self._ensure_collection()
        return self._connected

    def _ensure_collection(self) -> None:
        """collection 不存在则创建（幂等）"""
        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
            # 小节标题: FAISS 侧 metadata.json 一直有, Milvus 侧原先漏了,
            # 导致走 Milvus 检索时 section 恒为空、前端来源标注缺一截。
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields, description="智能客服知识库 chunk (P0)")
        self._collection = Collection(self._collection_name, schema)
        # 必需：创建向量索引后 search 才可用
        self._collection.create_index(
            "embedding",
            {"index_type": "AUTOINDEX", "metric_type": "COSINE", "params": {}},
        )
        self._collection.load()
        logger.info("Milvus collection 已创建并加载: %s", self._collection_name)

    @property
    def available(self) -> bool:
        """Milvus 是否可用（调用 _ensure_ready 触发首次连接）"""
        return self._ensure_ready()

    @property
    def is_loaded(self) -> bool:
        return self.available

    @property
    def total_count(self) -> int:
        if not self.available or self._collection is None:
            return 0
        try:
            return self._collection.num_entities
        except Exception as e:
            logger.warning("Milvus 计数失败: %s", e)
            return 0

    # ──────────────────────────────────────────────
    # 检索（接口对齐 FaissIndexManager.search）
    # ──────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list:
        """
        向量检索。query_embedding shape=(1, dim)。

        Returns: [{"content", "source", "score", "id", "document_id", "page", ...}]
          - score 为 COSINE 相似度（与 FAISS IndexFlatIP 归一化后等价）
          - id 为 Milvus 主键，供 MMR 的 get_vectors 回查
        """
        if not self.available or self._collection is None:
            return []
        try:
            col = Collection(self._collection_name)
            col.load()
            res = col.search(
                data=[np.asarray(query_embedding).reshape(1, -1).astype("float32").tolist()[0]],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["content", "source", "document_id", "page", "chunk_id", "section"],
            )
            results = []
            for hit in res[0]:
                entity = hit.entity  # pymilvus 2.4.x Hit.get 仅单参, 默认值用 or 兜底
                results.append({
                    "content": entity.get("content") or "",
                    "source": entity.get("source") or "",
                    "document_id": entity.get("document_id") or "",
                    "page": entity.get("page") or 1,
                    "chunk_id": entity.get("chunk_id") or "",
                    "section": entity.get("section") or "",
                    "id": int(hit.id),
                    "score": float(hit.score),
                })
            return results
        except Exception as e:
            logger.warning("Milvus 检索失败 (降级为空): %s", e)
            return []

    def batch_search(self, queries: np.ndarray, top_k: int = 10) -> list:
        """批量检索（兼容接口；小库单条循环即可）"""
        return [self.search(q.reshape(1, -1), top_k=top_k) for q in queries]

    # ──────────────────────────────────────────────
    # MMR 支持：按 id 回查向量
    # ──────────────────────────────────────────────

    def get_vectors(self, indices: list) -> np.ndarray:
        """
        按 Milvus 主键回查向量（供 MMR 多样性重排）。

        Returns: shape=(len(indices), dim)；缺失 id 跳过，空时返回 shape=(0,)
        """
        if not self.available or self._collection is None or not indices:
            return np.array([], dtype=np.float32)
        try:
            ids = [int(i) for i in indices]
            col = Collection(self._collection_name)
            res = col.query(expr=f"id in {ids}", output_fields=["embedding"])
            vec_by_id = {r["id"]: r["embedding"] for r in res}
            vectors = [vec_by_id[i] for i in ids if i in vec_by_id]
            if not vectors:
                return np.array([], dtype=np.float32)
            return np.asarray(vectors, dtype=np.float32)
        except Exception as e:
            logger.warning("Milvus 向量回查失败 (降级为空): %s", e)
            return np.array([], dtype=np.float32)

    # ──────────────────────────────────────────────
    # 写入 / 删除
    # ──────────────────────────────────────────────

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: list) -> None:
        """
        增量写入。metadata 字段与 collection schema 对齐
        （content/source/document_id/page/chunk_id/section）。
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"embeddings ({len(embeddings)}) 和 metadata ({len(metadata_list)}) 长度不匹配"
            )
        if not self.available or self._collection is None:
            logger.warning("Milvus 不可用, add_embeddings 跳过")
            return
        try:
            rows = []
            for emb, meta in zip(embeddings, metadata_list):
                row = {
                    "embedding": np.asarray(emb, dtype="float32").tolist(),
                    "content": meta.get("content", ""),
                    "source": meta.get("source", ""),
                    "document_id": meta.get("document_id", ""),
                    "page": int(meta.get("page", 1)),
                    "chunk_id": meta.get("chunk_id", ""),
                    "section": meta.get("section", ""),
                }
                rows.append(row)
            col = Collection(self._collection_name)
            col.insert(rows)
            col.flush()
            logger.info("Milvus 增量写入 %d 条", len(rows))
        except Exception as e:
            logger.error("Milvus 写入失败: %s", e)

    def delete_by_document(self, document_id: str) -> int:
        """删除某文档的所有向量（Milvus 原生硬删除）。返回删除条数。"""
        if not self.available or self._collection is None:
            return 0
        try:
            col = Collection(self._collection_name)
            col.flush()
            res = col.query(expr=f'document_id == "{document_id}"',
                            output_fields=["id"])
            if not res:
                return 0
            ids = [r["id"] for r in res]
            col.delete(expr=f"id in {ids}")
            col.flush()
            logger.info("Milvus 删除文档 %s: %d 条", document_id, len(ids))
            return len(ids)
        except Exception as e:
            logger.warning("Milvus 删除失败: %s", e)
            return 0

    # ──────────────────────────────────────────────
    # 管理
    # ──────────────────────────────────────────────

    def reload(self) -> None:
        """强制重新加载 collection"""
        with self._lock:
            self._check_done = False
            self._connected = False
            self._collection = None
        self._ensure_ready()

    def reset(self) -> None:
        """清空 collection（测试用）"""
        if self.available and self._collection is not None:
            try:
                self._collection.drop()
                self._collection = None
                self._check_done = False
                self._connected = False
                logger.info("Milvus collection 已删除: %s", self._collection_name)
            except Exception as e:
                logger.warning("Milvus reset 失败: %s", e)


# 全局单例
milvus_manager = MilvusManager()
