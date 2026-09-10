"""
智能客服 混合检索器

两路召回:
  Path A: 向量语义检索 (FAISS + bge-base-zh-v1.5) — 通过 FaissIndexManager 内存常驻
  Path B: BM25 稀疏检索 — 补充关键词盲区

融合: RRF (Reciprocal Rank Fusion) + MMR 多样性重排。
注: 知识图谱 (Neo4j) / 社区摘要 已在客服化改造中移除，只保留向量 + 稀疏两路。

性能优化:
  - FAISS 索引全局单例，避免重复加载
  - Redis 缓存层 (向量结果缓存)
  - bge-base-zh-v1.5 本地离线加载
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional, Any

# 强制 HuggingFace 离线 — 本机网络连不上 huggingface.co
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from dotenv import load_dotenv

from src.cache.faiss_manager import FaissIndexManager
from src.cache.redis_client import RedisCache
from src.retrieval.bm25_index import BM25Index, rrf_fusion

logger = logging.getLogger(__name__)

# 嵌入模型加载状态 (供系统状态查询)
EMBEDDING_MODEL_LOADED = False

# --- 核心：自动定位项目根目录 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)


class HybridRetriever:
    """两路混合检索器 (全局单例)"""
    _instance: Optional['HybridRetriever'] = None

    def __new__(cls) -> HybridRetriever:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        # 1. 向量索引管理器 (P0: Milvus 优先, FAISS 降级)
        from src.vectorstore.milvus_manager import MilvusManager
        milvus = MilvusManager()
        if milvus.available:
            logger.info("向量检索后端: Milvus (实体数 %d)", milvus.total_count)
            self.index_manager = milvus
        else:
            self.index_manager = FaissIndexManager()
            logger.info("向量检索后端: FAISS (Milvus 不可用降级)")

        # 2. Redis 缓存层 (优雅降级)
        self.cache = RedisCache()

        # 3. 加载 bge-base-zh-v1.5 模型 (进程级缓存, 惰性设备检测)
        self._model = None
        self._device = None  # 首次编码时检测（避免 torch.cuda 阻塞启动）

        # 4. BM25 稀疏检索 (可选, 索引不存在时不启用)
        self.bm25 = None
        self._init_bm25()

    def _init_bm25(self) -> None:
        """尝试加载 BM25 索引 (不阻塞, 不存在则静默跳过)"""
        from src.infra.paths import BM25_INDEX_PATH
        bm25_path = BM25_INDEX_PATH
        if os.path.exists(bm25_path):
            try:
                self.bm25 = BM25Index()
                self.bm25.load(bm25_path)
                logger.info("BM25 索引已加载: %s", bm25_path)
            except Exception as e:
                logger.warning("BM25 索引加载失败 (不影响主流程): %s", e)
                self.bm25 = None
        else:
            logger.info("BM25 索引不存在 (仅使用密集向量检索): %s", bm25_path)

    # ──────────────────────────────────────────────
    # 模型延迟加载
    # ──────────────────────────────────────────────

    @staticmethod
    def _find_local_model(cache_dir: str) -> str | None:
        """在本地缓存目录中查找已下载的 bge-base-zh-v1.5 模型"""
        candidates = [
            os.path.join(cache_dir, "models--BAAI--bge-base-zh-v1.5", "snapshots"),
            os.path.join(cache_dir, "BAAI", "bge-base-zh-v1.5"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                         "models--BAAI--bge-base-zh-v1.5", "snapshots"),
        ]
        for base in candidates:
            if not os.path.isdir(base):
                continue
            # 按快照目录名排序, 最新的优先
            snapshots = sorted(os.listdir(base), reverse=True)
            for snap in snapshots:
                snap_path = os.path.join(base, snap)
                if not os.path.isdir(snap_path):
                    continue
                files = set(os.listdir(snap_path))
                # 有效模型目录必须同时有 config.json 和权重文件
                has_config = "config.json" in files
                has_weights = ("pytorch_model.bin" in files or
                               "model.safetensors" in files)
                if has_config and has_weights:
                    return snap_path
        return None

    @property
    def model(self) -> Any:
        """延迟加载 bge-base-zh-v1.5 模型（首次编码时才加载）"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            cache_dir = os.getenv(
                "MODEL_CACHE_DIR",
                os.path.join(PROJECT_ROOT, "models")
            )
            logger.info("延迟加载 bge-base-zh-v1.5 模型...")
            # 惰性检测设备 — torch.cuda.is_available() 在 Windows 上可能耗时 50s
            if self._device is None:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            # 搜索本地缓存
            model_path = self._find_local_model(cache_dir)
            if model_path:
                logger.info("加载本地模型: %s", model_path)
                self._model = SentenceTransformer(model_path, device=self._device)
            else:
                logger.info("未找到本地模型, 尝试从 huggingface.co 下载 (cache=%s)", cache_dir)
                self._model = SentenceTransformer(
                    "BAAI/bge-base-zh-v1.5",
                    device=self._device,
                    cache_folder=cache_dir,
                )
        if self._model is not None:
            global EMBEDDING_MODEL_LOADED
            EMBEDDING_MODEL_LOADED = True
        return self._model

    def close(self) -> None:
        """释放在本对象中打开的资源"""
        if self.cache:
            self.cache.close()

    # ──────────────────────────────────────────────
    # Path A: 向量检索 (FAISS + MMR 多样性)
    # ──────────────────────────────────────────────

    @staticmethod
    def mmr_rerank(results: list, query_emb: np.ndarray,
                   index_manager, top_k: int = 5, lambd: float = 0.5) -> list:
        """
        Maximal Marginal Relevance (MMR) 多样性重排序

        MMR = λ * rel(q, d) - (1-λ) * max(sim(d, d_j))  (j ∈ selected)
        """
        if not results or len(results) <= 1:
            return results[:top_k]

        # 兼容 FAISS(faiss_idx) 与 Milvus(id) 两种结果键。
        # 注意 faiss_idx 可能为 0 (合法首条), 不能用 `or` 吞掉假值。
        indices = []
        for r in results:
            if "faiss_idx" in r and r.get("faiss_idx") is not None:
                indices.append(r["faiss_idx"])
            elif r.get("id") is not None:
                indices.append(r["id"])
        if not indices:
            return results[:top_k]

        doc_vectors = index_manager.get_vectors(indices)
        if doc_vectors.size == 0:
            return results[:top_k]

        qv = query_emb.astype(np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

        rel_scores = np.dot(doc_vectors, qv.T).flatten()

        n = len(results)
        selected = set()
        selected_list = []

        for _ in range(min(top_k, n)):
            best_score = -float("inf")
            best_idx = -1

            for i in range(n):
                if i in selected:
                    continue
                mmr_rel = rel_scores[i]
                mmr_div = 0.0
                if selected:
                    sim_to_selected = np.dot(
                        doc_vectors[i].reshape(1, -1),
                        doc_vectors[list(selected)].T,
                    ).flatten()
                    mmr_div = float(np.max(sim_to_selected))
                mmr_score = lambd * mmr_rel - (1 - lambd) * mmr_div
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.add(best_idx)
                selected_list.append(results[best_idx])

        return selected_list

    def vector_search(self, query: str, top_k: int = 5,
                       enable_mmr: bool = True) -> list:
        """向量语义检索 (带缓存)"""
        query_emb = self.model.encode([query], normalize_embeddings=True)

        emb_bytes = query_emb.tobytes()
        emb_hash = str(hash(emb_bytes) & 0xFFFFFFFF)

        cached = self.cache.get_vector_results(emb_hash, top_k)
        if cached is not None:
            logger.debug(f"向量缓存命中: {query[:30]}...")
            return cached

        mmr_top_k = max(top_k * 3, 15) if enable_mmr else top_k
        results = self.index_manager.search(query_emb, top_k=mmr_top_k)

        if enable_mmr and len(results) > top_k:
            results = self.mmr_rerank(
                results, query_emb, self.index_manager,
                top_k=top_k, lambd=0.5,
            )
        else:
            results = results[:top_k]

        # 过滤不含 content 的条目 (旧索引遗留问题)
        results = [r for r in results if r.get('content')]
        if len(results) < top_k:
            extra = self.index_manager.search(query_emb, top_k=top_k * 5)
            extra = [r for r in extra if r.get('content') and r not in results]
            results.extend(extra[:top_k - len(results)])

        if results:
            self.cache.set_vector_results(emb_hash, top_k, results, ttl=300)

        return results

    # ──────────────────────────────────────────────
    # Path B: BM25 稀疏检索
    # ──────────────────────────────────────────────

    def bm25_search(self, query: str, top_k: int = 10) -> list:
        """BM25 稀疏检索 (关键词匹配, 补充密集向量的盲区)"""
        if self.bm25 is None:
            return []
        try:
            return self.bm25.search(query, top_k=top_k)
        except Exception as e:
            logger.warning("BM25 检索失败: %s", e)
            return []

    # ──────────────────────────────────────────────
    # 混合检索主接口 (向量 + BM25)
    # ──────────────────────────────────────────────

    def _graph_retrieve(self, query: str) -> dict:
        """图谱多跳检索 (GraphRAG 局部; Neo4j 可用走 Neo4j, 否则内存降级)"""
        gr = getattr(self, "_graph_retriever", None)
        if gr is None:
            from src.knowledge_graph.retriever import GraphRetriever
            gr = GraphRetriever()
            self._graph_retriever = gr
        try:
            return gr.retrieve(query)
        except Exception as e:
            logger.debug("图谱检索失败(忽略): %s", e)
            return {"entries": [], "entities": []}

    def hybrid_retrieve(
        self, query: str, entity_hint: Optional[str] = None, top_k: int = 5
    ) -> dict:
        """
        两路混合检索: 向量 + BM25 → RRF 融合

        Returns:
            {"local": [...], "global": [], "graph": []}
            global/graph 保留为空 (客服场景不使用图谱/社区, 兼容下游上下文读取)
        """
        logger.info(f"混合检索: {query[:60]}...")

        # Path A: 密集向量检索
        dense_results = self.vector_search(query, top_k=top_k * 3)

        # Path B: BM25 稀疏检索 (补充密集检索的盲区)
        sparse_results = self.bm25_search(query, top_k=top_k * 3)

        # RRF 融合: 合并密集 + 稀疏结果
        if sparse_results:
            local_context = rrf_fusion(dense_results, sparse_results, top_k=top_k)
        else:
            local_context = dense_results[:top_k]

        # 图谱多跳补充 (GraphRAG): 命中实体/多跳语义时注入 1~3 条路径证据进 local,
        # 并整体保留在 graph 位 (供前端推理路径/评测用)
        graph_result = self._graph_retrieve(query)
        graph_context = list(graph_result.get("entries", [])) if graph_result.get("entries") else []
        # 仅在关系类/多实体查询注入 prompt (推荐/罗列等单实体查询不做图谱增强, 避免噪音)
        from src.knowledge_graph.retriever import RELATIONAL_HINTS
        relational = len(graph_result.get("entities", [])) >= 2 or any(
            k in query for k in RELATIONAL_HINTS)
        if graph_context and relational:
            existing_srcs = {c.get("source") for c in local_context}
            added = [e for e in graph_context if e.get("source") not in existing_srcs]
            local_context = list(local_context) + added[:3]

        return {
            "local": local_context,
            "global": [],
            "graph": graph_context,
        }


# 兼容旧代码的便捷函数
def create_retriever() -> HybridRetriever:
    return HybridRetriever()
