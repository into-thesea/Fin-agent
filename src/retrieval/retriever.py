"""
智能客服 混合检索器

两路召回:
  Path A: 向量语义检索 (Milvus + bge-base-zh-v1.5) — 唯一向量后端, 无降级
  Path B: BM25 稀疏检索 — 补充关键词盲区

融合: 稠密路内先做 MMR 多样性重排, 再与稀疏路做 RRF (Reciprocal Rank Fusion)。
图谱: Neo4j 为运行时唯一图谱后端 (不降级内存图), 关系类/多实体查询经 _graph_retrieve
      注入 1~3 条多跳路径证据, 并保留在 graph 位供前端画推理路径与评测使用。

性能优化:
  - Redis 缓存层 (向量结果缓存)
  - bge-base-zh-v1.5 本地离线加载
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

# 强制 HuggingFace 离线 — 本机网络连不上 huggingface.co
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 设备引导: 默认强制CPU。Windows首次 torch.cuda.is_available() 探测CUDA驱动实测14s+;
# import torch 前设 CUDA_VISIBLE_DEVICES=-1, 跳过CUDA库加载, bge冷启动降到约0.3s。
# 需要GPU: .env 设 EMBED_DEVICE=cuda (不写入-1, 允许探测)。
from dotenv import load_dotenv as _load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_load_dotenv(os.path.join(_ROOT, ".env"), override=False)
if os.getenv("EMBED_DEVICE", "cpu").strip().lower() == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
from dotenv import load_dotenv

from src.cache.redis_client import RedisCache
from src.retrieval.bm25_index import BM25Index, rrf_fusion

logger = logging.getLogger(__name__)

# ── BM25 金丝雀 ──────────────────────────────
# 判定「索引/分词坏了」在索引加载后验证一次, 而不是每次查询猜。
# 原因: 稠密路(ANN)永远返回 k 个最近邻, 所以「稠密有条 + 稀疏 0 条」是有歧义的 ——
# 既可能是 BM25 挂了, 也可能只是 query 跟语料零词面重叠(离题问题如「给我写一首诗」)。
# 前者必须响亮失败(2026-09-13「假混合检索」缺陷), 后者应正常兜底而不是给用户 500。
_BM25_CANARY_QUERY = "存款保险"


def bm25_canary_ok(bm25) -> bool:
    """索引健康度金丝雀 —— 每个索引对象只验一次(结果挂在索引上, 索引重建自然失效)"""
    cached = getattr(bm25, "_canary_ok", None)
    if cached is not None:
        return cached
    ok = False
    if getattr(bm25, "nd", 0):
        try:
            ok = bool(bm25.search(_BM25_CANARY_QUERY, top_k=1))
        except Exception as e:
            logger.error("BM25 金丝雀探针异常: %s", e)
    bm25._canary_ok = ok
    if not ok:
        logger.error("BM25 金丝雀探针 %r 返回 0 条 —— 索引为空或分词已失效",
                     _BM25_CANARY_QUERY)
    return ok


def format_graph_context(items) -> str:
    """把 contexts['graph'] 渲染成可注入 Prompt 的文本。

    graph 位刻意保留结构化路径条目 ({"content","source","path"}) —— 前端画推理路径、
    评测算 path_structure_consistency 都依赖它, 所以不能在生产端降级成字符串。
    但消费端有 9 处历史写法是直接 "\\n".join(contexts['graph']): 图谱没命中时是空列表
    不报错, **一旦命中就 TypeError: expected str instance, dict found**。
    统一走这里, 别在各处再手写 join。
    """
    lines = []
    for it in items or []:
        if isinstance(it, str):
            lines.append(it)
            continue
        it = it or {}
        src = it.get("source", "知识图谱")
        lines.append(f"- {it.get('content', '')} (来源: {src})")
    return "\n".join(lines)

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
        import threading
        self._model = None
        self._device = None  # 由 EMBED_DEVICE 决定; cpu 时不做 cuda 探测
        self._model_lock = threading.Lock()  # 双检锁: 预热与早期请求不重复加载

        # 4. BM25 稀疏检索
        self.bm25 = None
        self._init_bm25()

        # 5. 重排器 —— 由可编辑配置决定, 惰性构建 (见 _reranker_for)
        from src.config import settings as _s  # 仅用于首次启动提示
        self.reranker = None
        self._reranker_model = ""
        self.rerank_pool = 0
        if _s.rerank_enabled:
            logger.info("检测到 RERANK_ENABLED=true（.env）; 线上以 data/kb_settings.json 为准")

        # 6. 父子文档分块: parent_id → parent chunk 映射 (懒加载, 首次检索时构建)
        self._parent_map = None

    def _reranker_for(self, cfg: dict):
        """按配置取重排器。配置里关掉了就返回 None。

        按模型名缓存实例 —— 每个请求都新建 ApiReranker 会白跑构造校验;
        而配置改了模型名必须能换, 所以用 _reranker_model 记住当前实例对应哪个模型。
        """
        if not cfg.get("rerank_enabled"):
            return None
        model = cfg.get("rerank_model") or "gte-rerank-v2"
        if self.reranker is None or self._reranker_model != model:
            from src.retrieval.reranker import ApiReranker
            self.reranker = ApiReranker(model=model)
            self._reranker_model = model
            logger.info("重排器已装载: %s", model)
        return self.reranker

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
            with self._model_lock:
                if self._model is not None:      # 双检: 等待期间可能已被预热加载
                    return self._model
                from sentence_transformers import SentenceTransformer
                cache_dir = os.getenv(
                    "MODEL_CACHE_DIR",
                    os.path.join(PROJECT_ROOT, "models")
                )
                logger.info("延迟加载 bge-base-zh-v1.5 模型...")
                # 设备: 尊重 EMBED_DEVICE。cpu(默认)直接用, 不触发 Windows 上极慢的
                # torch.cuda.is_available() 探测(实测14s+); 仅显式 cuda/auto 才探测。
                if self._device is None:
                    pref = os.getenv("EMBED_DEVICE", "cpu").strip().lower()
                    if pref == "cpu":
                        self._device = "cpu"
                    else:
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
    # Path A: 向量检索 (Milvus + MMR 多样性)
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

        # Milvus 结果键为 id (主键), 供 index_manager.get_vectors 回查。
        # 必须与 results 一一对位 —— 下方按位置 i 索引 doc_vectors, 少了会越界。
        indices = [r.get("id") for r in results]
        if any(i is None for i in indices):
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
                       enable_mmr: bool = None) -> list:
        """向量语义检索 (带缓存)。MMR 开关与 λ 来自可编辑配置。"""
        from src.core.kb_settings import get_retrieval
        cfg = get_retrieval()
        if enable_mmr is None:
            enable_mmr = bool(cfg.get("mmr_enabled", True))
        mmr_lambda = float(cfg.get("mmr_lambda", 0.5))

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
                top_k=top_k, lambd=mmr_lambda,
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


    def _load_parent_map(self) -> dict:
        """从分块文件加载所有父块, 构建 chunk_id → parent chunk 映射 (懒加载, 进程内缓存)"""
        if self._parent_map is not None:
            return self._parent_map
        import json as _json

        from src.infra.paths import CHUNKS_PROCESSED_PATH
        parent_map = {}
        try:
            with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        c = _json.loads(line)
                        if c.get("chunk_type") == "parent":
                            parent_map[c["chunk_id"]] = c
        except FileNotFoundError:
            logger.warning("分块文件不存在, 父块映射为空")
        self._parent_map = parent_map
        logger.info("加载父块映射: %d 个父块", len(parent_map))
        return parent_map

    def _expand_to_parents(self, chunks: list) -> list:
        """子块命中后展开为父块完整内容, 同一父块去重

        父子文档分块的核心: 用小chunk精准检索, 返回父章节完整上下文给LLM。
        没有parent_id的chunk (如上传文档的chunk) 原样保留。
        """
        parent_map = self._load_parent_map()
        result = []
        seen_parent_ids = set()
        for c in chunks:
            parent_id = c.get("parent_id")
            if parent_id and parent_id in parent_map:
                if parent_id in seen_parent_ids:
                    continue  # 同一父块已展开过, 去重
                seen_parent_ids.add(parent_id)
                result.append(parent_map[parent_id])
            else:
                # 无parent_id (上传文档或旧格式chunk), 原样保留
                pid = c.get("chunk_id")
                if pid not in seen_parent_ids:
                    seen_parent_ids.add(pid)
                    result.append(c)
        return result

    def hybrid_retrieve(
        self, query: str, entity_hint: Optional[str] = None, top_k: Optional[int] = None
    ) -> dict:
        """
        两路混合检索: 向量 + BM25 → RRF 融合 (+ 可选精排 + 图谱补充)

        Args:
            entity_hint: **接受但当前不使用**。原打算用它做「按产品收敛检索范围」的
                元数据过滤, 2026-09-14 实测后放弃, 原因见下 —— 保留参数只为不改动
                6 处调用方签名。
            top_k: 不传(None)时取可编辑配置(data/kb_settings.json)的 top_k。
                默认值不能在函数签名里求值 —— 那是定义时求一次, 配置改了不生效。
                显式传参的调用方(工具/评测脚本)行为不变。

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

        from src.core.kb_settings import get_retrieval
        cfg = get_retrieval()
        if top_k is None:
            top_k = int(cfg.get("top_k", 5))
        reranker = self._reranker_for(cfg)
        # 候选池: 启用重排时先多取, 精排后再收敛到 top_k
        # max(...,1): 配置文件是手工可改的, 校验只在 API 那一侧 —— 池子为 0 会让
        # 两路检索都取 0 条, 变成"检索成功但什么都没有"
        pool_k = max(int(cfg.get("rerank_pool", 20)), 1) if reranker is not None else top_k

        # Path A: 密集向量检索
        dense_results = self.vector_search(query, top_k=pool_k * 3)

        # Path B: BM25 稀疏检索 (补充密集检索的盲区)
        sparse_results = self.bm25_search(query, top_k=pool_k * 3)

        # RRF 融合: 两路都必须有结果。
        # 原先 `if sparse_results: RRF else: dense[:top_k]` 是个静默降级 ——
        # 它把「BM25 索引是空的 / 分词失效」伪装成「本来就没关键词命中」,
        # 让纯向量检索冒充混合检索(2026-09-13 的「假混合检索」缺陷)。
        #
        # 但「索引坏了」和「这条 query 没有关键词命中」是两件事, 必须分开:
        # 离题/超纲问题(如「给我写一首诗」)在健康索引上本来就该 0 命中,
        # 一律 raise 会把正常提问变成 500。用金丝雀探针判定索引健康度 ——
        # 索引坏了(探针 0 条)照旧响亮失败, 索引健康则只记警告并退回稠密路。
        if not sparse_results:
            if not bm25_canary_ok(self.bm25):
                raise RuntimeError(
                    "BM25 稀疏路返回 0 条 —— 混合检索无法成立(纯向量不等于混合检索)。"
                    "请检查 BM25 索引是否为空或分词是否失效。"
                )
            logger.warning(
                "BM25 稀疏路无关键词命中 (索引 %d 篇, query=%r) → 本轮退化为纯向量",
                self.bm25.nd, query[:40],
            )
        local_context = rrf_fusion(dense_results, sparse_results,
                                   top_k=pool_k, k=int(cfg.get("rrf_k", 60)))

        # 精排: 用语义重排模型把候选池收敛到 top_k
        rerank_failed = None
        if reranker is not None:
            try:
                local_context = reranker.rerank(query, local_context, top_k=top_k)
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

        # 父子文档分块: 子块检索命中后展开为父块完整内容 (同一父块去重)
        local_context = self._expand_to_parents(local_context)

        return {
            "local": local_context,
            "global": [],
            "graph": graph_context,
            # 重排异常时非 None —— 让「重排没生效」可被上层/评测看见,
            # 而不是和「重排后顺序恰好没变」混在一起
            "rerank_failed": rerank_failed,
            # 稀疏路 0 命中(索引健康但 query 离题)时同样留痕, 与 09-13 修的
            # 「假混合检索」区分开: 那个是索引坏的静默降级, 这个是离题问题的正常退化
            "sparse_empty": not sparse_results,
        }

    # ──────────────────────────────────────────────
    # 调试视图: 各通路中间结果 (调试台用, 不在线上热路径)
    # ──────────────────────────────────────────────

    def retrieve_debug(self, query: str, top_k: int = 5) -> dict:
        """返回向量/稀疏/融合/图谱各路明细, 用于"为什么这条没搜到"的排查。

        独立于 hybrid_retrieve: 后者是线上热路径, 返回值被 agent/评测/E2E 多处消费,
        为调试加字段会扩大回归面。

        融合用的 rrf_k 必须与 hybrid_retrieve 一样取自配置 —— 否则调试台显示的排序
        会和线上实际排序不一致, 而"如实反映线上发生了什么"正是这个视图的全部价值。
        """
        import time as _t

        from src.core.kb_settings import get_retrieval
        from src.retrieval.bm25_index import rrf_fusion

        cfg = get_retrieval()
        t0 = _t.perf_counter()

        dense = self.vector_search(query, top_k=top_k * 3)
        sparse = self.bm25_search(query, top_k=top_k * 3)

        sparse = sparse or []
        fused = (rrf_fusion(dense, sparse, top_k=top_k, k=int(cfg.get("rrf_k", 60)))
                 if sparse else list(dense[:top_k]))

        dense_rank = {c.get("chunk_id"): i + 1 for i, c in enumerate(dense)}
        sparse_rank = {c.get("chunk_id"): i + 1 for i, c in enumerate(sparse)}

        fused_out = []
        for i, c in enumerate(fused):
            cid = c.get("chunk_id")
            found = []
            if cid in dense_rank:
                found.append("dense")
            if cid in sparse_rank:
                found.append("sparse")
            fused_out.append({
                "chunk_id": cid,
                "source": c.get("source", ""),
                "section": c.get("section", ""),
                "rank": i + 1,
                "rrf_score": c.get("rrf_score"),
                "dense_rank": dense_rank.get(cid),
                "sparse_rank": sparse_rank.get(cid),
                "rerank_score": c.get("rerank_score"),
                "found_by": found,
            })

        try:
            graph = self._graph_retrieve(query)
        except Exception as e:
            # 调试台要把失败显示出来, 而不是假装图谱没命中
            graph = {"entries": [], "entities": [], "error": str(e)}

        return {
            "dense": [{"chunk_id": c.get("chunk_id"), "source": c.get("source", ""),
                       "rank": i + 1, "score": c.get("score")}
                      for i, c in enumerate(dense)],
            "sparse": [{"chunk_id": c.get("chunk_id"), "source": c.get("source", ""),
                        "rank": i + 1, "score": c.get("score")}
                       for i, c in enumerate(sparse)],
            "fused": fused_out,
            "graph": graph,
            "config_used": {**cfg, "top_k": top_k},
            "elapsed_ms": round((_t.perf_counter() - t0) * 1000, 1),
        }


# 兼容旧代码的便捷函数
def create_retriever() -> HybridRetriever:
    return HybridRetriever()
