"""
智能客服 混合检索器

两路召回:
  Path A: 向量语义检索 (Milvus + bge-base-zh-v1.5) — 唯一向量后端, 无降级
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
        # 1. 向量检索后端: Milvus (唯一)
        #    刻意保留 FAISS 降级 —— 两种后端会给出不同答案(不同机器结果不同),
        #    且历史遗留的 FAISS 路径上传的新文档根本进不去, 会在 Milvus 环境下
        #    检索不到。静默降级 = 用户拿到错误的、不完整的检索结果却毫无提示。
        #    构造时不抛错(让服务能起来、/health 能如实报告), 检索时直接失败。
        from src.vectorstore.milvus_manager import MilvusManager
        self.index_manager = MilvusManager()
        if self.index_manager.available:
            logger.info("向量检索后端: Milvus (实体数 %d)", self.index_manager.total_count)
        else:
            logger.error(
                "Milvus 不可用 —— 向量检索将直接报错, 不再降级 FAISS。"
                "请启动 Milvus (docker compose up -d milvus)。"
            )

        # 2. Redis 缓存层 (缓存失效只影响速度, 不影响答案 —— 可降级)
        self.cache = RedisCache()

        # 3. 加载 bge-base-zh-v1.5 模型 (进程级缓存, 惰性设备检测)
        self._model = None
        self._device = None  # 首次编码时检测（避免 torch.cuda 阻塞启动）

        # 4. BM25 稀疏检索
        self.bm25 = None
        self._init_bm25()

        # 5. 重排器 (默认关闭; 开启后先超取候选池再精排)
        from src.config import settings
        self.reranker = None
        self.rerank_pool = 0
        if settings.rerank_enabled:
            from src.retrieval.reranker import ApiReranker
            self.reranker = ApiReranker(model=settings.rerank_model)
            self.rerank_pool = max(int(settings.rerank_pool), 1)
            logger.info("重排已启用: %s (候选池 %d)", settings.rerank_model, self.rerank_pool)

    def _init_bm25(self) -> None:
        """加载 BM25 索引。

        「索引不存在」不是可忽略的小事 —— 它意味着混合检索退化成了纯向量检索,
        而这正是 2026-09-13 修掉的「假混合检索」缺陷。加载失败直接抛错。
        """
        from src.infra.paths import BM25_INDEX_PATH
        if not os.path.exists(BM25_INDEX_PATH):
            logger.error(
                "BM25 索引不存在: %s —— 混合检索的稀疏路不可用, 检索将直接报错。"
                "请运行 scripts/sync_kb.py 重建索引。", BM25_INDEX_PATH,
            )
            self.bm25 = None
            return
        self.bm25 = BM25Index()
        self.bm25.load(BM25_INDEX_PATH)
        logger.info("BM25 索引已加载: %s", BM25_INDEX_PATH)

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
        """BM25 稀疏检索 (关键词匹配, 补充密集向量的盲区)

        不再 `except → return []`: 吞掉异常会让上游以为「稀疏路没结果」,
        从而静默退化成纯向量检索。
        """
        if self.bm25 is None:
            raise RuntimeError(
                "BM25 索引未加载 —— 混合检索的稀疏路不可用。"
                "请运行 scripts/sync_kb.py 重建索引。"
            )
        return self.bm25.search(query, top_k=top_k)

    # ──────────────────────────────────────────────
    # 混合检索主接口 (向量 + BM25)
    # ──────────────────────────────────────────────

    def _graph_retrieve(self, query: str) -> dict:
        """图谱多跳检索 (GraphRAG 局部, Neo4j 唯一后端)

        不再吞异常: 关系类查询依赖图谱证据, 吞掉等于悄悄少了一部分依据。
        调用方 hybrid_retrieve 按查询类型决定「失败」还是「本次不增强」。
        """
        gr = getattr(self, "_graph_retriever", None)
        if gr is None:
            from src.knowledge_graph.retriever import GraphRetriever
            gr = GraphRetriever()
            self._graph_retriever = gr
        return gr.retrieve(query)

    def hybrid_retrieve(
        self, query: str, entity_hint: Optional[str] = None, top_k: int = 5
    ) -> dict:
        """
        两路混合检索: 向量 + BM25 → RRF 融合 (+ 可选精排 + 图谱补充)

        Args:
            entity_hint: **接受但当前不使用**。原打算用它做「按产品收敛检索范围」的
                元数据过滤, 2026-09-14 实测后放弃, 原因见下 —— 保留参数只为不改动
                6 处调用方签名。

        为什么不做「按产品过滤」的元数据过滤(实测数据, 别再重做一遍):
          1. 触发源是死的: RouterAgent.get_entity_hint() 恒返回 None, 该参数永远是 None。
          2. 从查询本身识别产品名做硬过滤会**滤掉正确答案**: golden 里 13 个提到产品名
             的问题中, 7 个(54%)的答案证据在产品文档之外(如 deposit_insurance.md /
             risk_level.md / agency_products.md)。理财域里查询中的产品名标识的是
             「问的是谁」, 不是「答案在哪」—— 「大额存单受存款保险保障吗」必须同时要
             产品文档与存款保险制度文档。
          3. 收益上限很低: 基线 Recall@5 已 97.5%, 只有 3/40 的问题的 golden 被
             「别的产品的 chunk」压在下面。
          4. 真正按元数据筛选的场景(按风险等级/产品类型选产品)已有更好的实现:
             tools 里的 query_products(product_type, risk, keyword), 走 catalog
             结构化查询 —— 确定性、零误伤, 不该用向量检索的过滤开关来替代。

        Returns:
            {"local": [...], "global": [], "graph": [...], "rerank_failed": str|None}
        """
        logger.info(f"混合检索: {query[:60]}...")

        # 候选池: 启用重排时先多取, 精排后再收敛到 top_k
        pool_k = self.rerank_pool if self.reranker is not None else top_k

        # Path A: 密集向量检索
        dense_results = self.vector_search(query, top_k=pool_k * 3)

        # Path B: BM25 稀疏检索 (补充密集检索的盲区)
        sparse_results = self.bm25_search(query, top_k=pool_k * 3)

        # RRF 融合: 两路都必须有结果。
        # 原先 `if sparse_results: RRF else: dense[:top_k]` 是个静默降级 ——
        # 它把「BM25 索引是空的 / 分词失效」伪装成「本来就没关键词命中」,
        # 让纯向量检索冒充混合检索(2026-09-13 的「假混合检索」缺陷)。
        if not sparse_results:
            raise RuntimeError(
                "BM25 稀疏路返回 0 条 —— 混合检索无法成立(纯向量不等于混合检索)。"
                "请检查 BM25 索引是否为空或分词是否失效。"
            )
        local_context = rrf_fusion(dense_results, sparse_results, top_k=pool_k)

        # 精排: 用语义重排模型把候选池收敛到 top_k
        rerank_failed = None
        if self.reranker is not None:
            try:
                local_context = self.reranker.rerank(query, local_context, top_k=top_k)
            except Exception as e:
                # 重排只影响「排序质量」, 不改变答案有没有依据 —— 失败时保留 RRF
                # 原序并大声记录, 不因为外部接口抖动就让整轮问答挂掉。
                # 这是与「检索后端不可用」不同性质的一类降级: 答案不会变错, 只会变糙。
                logger.error("重排失败, 本次沿用 RRF 原序: %s", e, exc_info=True)
                rerank_failed = str(e)
                local_context = local_context[:top_k]
        else:
            local_context = local_context[:top_k]

        # 图谱多跳补充 (GraphRAG): 命中实体/多跳语义时注入 1~3 条路径证据进 local,
        # 并整体保留在 graph 位 (供前端推理路径/评测用)
        from src.knowledge_graph.retriever import RELATIONAL_HINTS
        relational_hint = any(k in query for k in RELATIONAL_HINTS)
        try:
            graph_result = self._graph_retrieve(query)
        except Exception as e:
            # 关系类查询的答案依赖图谱证据 → 直接失败, 不给残缺答案;
            # 其他查询图谱只是可选增强(下面 relational 为假时本就不会注入), 记为故障但继续。
            if relational_hint:
                logger.error("图谱检索失败且该查询依赖图谱证据: %s", e, exc_info=True)
                raise
            logger.error("图谱检索失败 (非关系类查询, 本次无图谱增强): %s", e, exc_info=True)
            graph_result = {"entries": [], "entities": [], "graph_error": str(e)}
        graph_context = list(graph_result.get("entries", [])) if graph_result.get("entries") else []
        # 仅在关系类/多实体查询注入 prompt (推荐/罗列等单实体查询不做图谱增强, 避免噪音)
        relational = len(graph_result.get("entities", [])) >= 2 or relational_hint
        if graph_context and relational:
            existing_srcs = {c.get("source") for c in local_context}
            added = [e for e in graph_context if e.get("source") not in existing_srcs]
            local_context = list(local_context) + added[:3]

        return {
            "local": local_context,
            "global": [],
            "graph": graph_context,
            # 重排异常时非 None —— 让「重排没生效」可被上层/评测看见,
            # 而不是和「重排后顺序恰好没变」混在一起
            "rerank_failed": rerank_failed,
        }


# 兼容旧代码的便捷函数
def create_retriever() -> HybridRetriever:
    return HybridRetriever()
