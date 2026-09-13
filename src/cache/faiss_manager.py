"""
Fin-Agent 向量索引管理器

核心改进:
  1. 内存常驻 — FAISS 索引和元数据在进程生命周期内只加载一次
  2. 读写锁 — 支持增量更新时并发读不阻塞
  3. 延迟加载 — 首次检索时才实际加载索引
  4. 线程安全 — 使用 threading.RLock 保证多线程安全
"""

import os
import json
import threading
import logging
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss 库未安装，向量索引管理器将以降级模式运行 (no-op)")

from src.infra.paths import FAISS_INDEX_DIR

DEFAULT_INDEX_DIR = FAISS_INDEX_DIR


class FaissIndexManager:
    """FAISS 索引管理器 (全局单例)"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        """线程安全的单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, index_dir: str = None):
        if getattr(self, "_initialized", False):
            return
        with self._lock:
            if self._initialized:
                return

        self._index_dir = index_dir or DEFAULT_INDEX_DIR
        self._index: Any = None
        self._metadata: list = []
        self._rw_lock = threading.RLock()  # 可重入读写锁
        self._dimension: int = 768  # bge-base-zh-v1.5 默认维度
        self._index_ntotal: int = 0
        self._loaded = False
        self._initialized = True
        logger.info(f"FaissIndexManager 已初始化, 索引目录: {self._index_dir}")

    # ──────────────────────────────────────────────
    # 索引加载
    # ──────────────────────────────────────────────

    def _ensure_loaded(self):
        """延迟加载索引（首次检索时触发）"""
        if self._loaded:
            return
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，索引加载跳过")
            self._loaded = True
            return
        with self._rw_lock:
            if self._loaded:
                return

            index_path = os.path.join(self._index_dir, "index.faiss")
            meta_path = os.path.join(self._index_dir, "metadata.json")

            if not os.path.exists(index_path) or not os.path.exists(meta_path):
                logger.warning(f"FAISS 索引文件不存在: {index_path}")
                self._index = None
                self._metadata = []
                self._loaded = True
                return

            try:
                self._index = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                self._index_ntotal = self._index.ntotal
                self._loaded = True
                logger.info(
                    f"FAISS 索引已加载: {self._index_ntotal} 条向量, "
                    f"维度 {self._index.d if hasattr(self._index, 'd') else 'N/A'}"
                )
            except Exception as e:
                logger.error(f"FAISS 索引加载失败: {e}")
                self._index = None
                self._metadata = []
                self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._index is not None

    @property
    def total_count(self) -> int:
        return self._index_ntotal if self._index else 0

    # ──────────────────────────────────────────────
    # 向量检索 (并发读)
    # ──────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list:
        """
        执行向量检索（线程安全，支持并发读）

        Args:
            query_embedding: shape=(1, dim) 的归一化向量
            top_k: 返回 TOP-K 结果

        Returns:
            [{"content": str, "source": str, "score": float, ...}, ...]
        """
        self._ensure_loaded()
        if self._index is None:
            return []

        with self._rw_lock:
            distances, indices = self._index.search(
                query_embedding.astype("float32"), top_k
            )

        results = []
        with self._rw_lock:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self._metadata):
                    meta = self._metadata[idx]
                    # 已删除条目必须跳过 —— 否则删除只改了标记, 检索照样召回
                    if meta.get("deleted"):
                        continue
                    meta = meta.copy()
                    meta["score"] = float(distances[0][i])
                    meta["faiss_idx"] = int(idx)  # 用于 MMR 向量重构
                    results.append(meta)

        return results

    def get_vectors(self, indices: list) -> np.ndarray:
        """
        从 FAISS 索引中还原指定位置的向量 (用于 MMR 多样性重排序)

        Args:
            indices: 向量在索引中的位置列表

        Returns:
            shape=(len(indices), dim) 的向量矩阵
        """
        self._ensure_loaded()
        if self._index is None or not indices:
            return np.array([], dtype=np.float32)

        valid_indices = [i for i in indices if 0 <= i < self._index.ntotal]
        if not valid_indices:
            return np.array([], dtype=np.float32)

        with self._rw_lock:
            vectors = np.vstack([
                self._index.reconstruct(int(i)).reshape(1, -1)
                for i in valid_indices
            ])
        return vectors

    def batch_search(self, queries: np.ndarray, top_k: int = 10) -> list:
        """批量检索（减少 GIL 切换开销）"""
        self._ensure_loaded()
        if self._index is None:
            return [[] for _ in range(len(queries))]

        with self._rw_lock:
            distances, indices = self._index.search(
                queries.astype("float32"), top_k
            )

        all_results = []
        with self._rw_lock:
            for q_idx in range(len(queries)):
                q_results = []
                for i, idx in enumerate(indices[q_idx]):
                    if 0 <= idx < len(self._metadata):
                        meta = self._metadata[idx]
                        if meta.get("deleted"):
                            continue  # 同上: 已删除条目不召回
                        meta = meta.copy()
                        meta["score"] = float(distances[q_idx][i])
                        q_results.append(meta)
                all_results.append(q_results)

        return all_results

    # ──────────────────────────────────────────────
    # 增量更新 (写锁互斥)
    # ──────────────────────────────────────────────

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: list):
        """
        增量添加向量（线程安全，写入时阻塞读取）

        Args:
            embeddings: shape=(N, dim) 的归一化向量
            metadata_list: 长度 N 的元数据列表
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"embeddings ({len(embeddings)}) 和 metadata ({len(metadata_list)}) 长度不匹配"
            )
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，跳过索引更新")
            return

        with self._rw_lock:  # 写锁 — 阻塞所有读
            # 必须先确保已从磁盘加载。否则新进程的首次 add_embeddings 会看到
            # _index is None 而新建一个空索引, 落盘时把已有的 index.faiss
            # **整个覆盖**掉 —— 实测: 知识库重建出的 213 条向量, 上传一次文档后
            # 磁盘上只剩该文档的 2 条。
            self._ensure_loaded()
            if self._index is None:
                # 索引文件不存在或为空 → 真正的首次创建
                self._dimension = embeddings.shape[1]
                self._index = faiss.IndexFlatIP(self._dimension)
                self._metadata = []

            self._index.add(np.array(embeddings).astype("float32"))
            self._metadata.extend(metadata_list)
            self._index_ntotal = self._index.ntotal

            # 持久化
            self._persist()

        logger.info(
            f"索引增量更新完成: +{len(embeddings)} 条, "
            f"总数: {self._index_ntotal}"
        )

    def delete_by_document(self, document_id: str):
        """
        标记删除某个文档的所有向量。
        FAISS 不支持原地删除，策略：
          1. 在 metadata 中将目标文档的条目标记为 deleted
          2. 当 deleted 比例 > 30% 时触发重建

        Returns: 被标记删除的条目数
        """
        with self._rw_lock:
            count = 0
            for meta in self._metadata:
                if meta.get("document_id") == document_id and not meta.get("deleted"):
                    meta["deleted"] = True
                    count += 1

            if count > 0:
                deleted_ratio = sum(
                    1 for m in self._metadata if m.get("deleted")
                ) / max(len(self._metadata), 1)

                if deleted_ratio > 0.3:
                    logger.info(
                        f"删除比例 {deleted_ratio:.1%} > 30%, 触发索引重建"
                    )
                    self._rebuild()
                else:
                    self._persist_metadata()

        return count

    def _rebuild(self):
        """重建索引（过滤已删除条目）

        必须**同时**重建向量索引与 metadata。原实现只重写 metadata.json 就把
        索引置空重载, 而磁盘上的 index.faiss 仍是旧的、向量数不变 —— 于是
        metadata 的第 i 项不再对应索引的第 i 条向量, 检索命中会配上错误的内容
        与归属(引用张冠李戴), 且尾部合法向量被 `idx < len(metadata)` 静默丢弃。
        """
        if self._index is None:
            return
        active = [(i, m) for i, m in enumerate(self._metadata) if not m.get("deleted")]

        old_index = self._index
        if not active:
            self._index = None
            self._metadata = []
            self._index_ntotal = 0
            faiss.write_index(old_index, os.path.join(self._index_dir, "index.faiss"))
            self._persist_metadata()
            logger.warning("重建后索引为空")
            return

        # 按保留条目取回向量, 建一个全新的连续索引 —— 下标与 metadata 重新对齐
        vectors = np.vstack([
            old_index.reconstruct(int(i)).reshape(1, -1) for i, _ in active
        ]).astype("float32")
        new_index = faiss.IndexFlatIP(old_index.d)
        new_index.add(vectors)

        self._index = new_index
        self._metadata = [m for _, m in active]
        self._index_ntotal = new_index.ntotal
        self._persist()
        logger.info("索引重建完成: %d 条有效向量 (索引与 metadata 已同步)", len(self._metadata))

    # ──────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────

    def _persist(self):
        """将当前索引写入磁盘"""
        if self._index is None or not FAISS_AVAILABLE:
            return
        os.makedirs(self._index_dir, exist_ok=True)
        faiss.write_index(
            self._index,
            os.path.join(self._index_dir, "index.faiss")
        )
        self._persist_metadata()

    def _persist_metadata(self):
        """将元数据写入磁盘"""
        meta_path = os.path.join(self._index_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────
    # 重置
    # ──────────────────────────────────────────────

    def replace_all(self):
        """丢弃当前索引, 从空索引重新开始 (全量重建用)。

        置 `_loaded = True` 是关键 —— 不置的话后续 add_embeddings 里的
        `_ensure_loaded()` 会把磁盘上的旧索引再读回来, 新数据追加在旧数据后面。
        """
        with self._rw_lock:
            self._index = None
            self._metadata = []
            self._index_ntotal = 0
            self._loaded = True
        logger.info("FAISS 索引已清空, 准备全量重建")

    def reset(self):
        """清空并重置索引（用于测试）"""
        with self._rw_lock:
            self._index = None
            self._metadata = []
            self._index_ntotal = 0
            self._loaded = False
        logger.info("FAISS 索引已重置")

    def reload(self):
        """强制从磁盘重新加载"""
        with self._rw_lock:
            self._loaded = False
            self._index = None
        self._ensure_loaded()


# 全局单例实例
faiss_index_manager = FaissIndexManager()
faiss_manager = faiss_index_manager  # 兼容旧 import 名
