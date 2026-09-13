"""
Fin-Agent 向量索引构建器 (企业版)

支持:
  - 增量写入 Milvus 向量库 (唯一向量后端)
  - 文档级元数据追踪 (document_id, page, source)
"""

import os
import json
import logging
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# --- 核心：自动定位项目根目录 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.join(PROJECT_ROOT, "models"))

# 模型单例 (进程级别)
_model_instance = None
_model_device = None


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
            has_config = "config.json" in files
            has_weights = ("pytorch_model.bin" in files or
                           "model.safetensors" in files)
            if has_config and has_weights:
                return snap_path
    return None


def _get_model():
    """延迟加载 bge-base-zh-v1.5 模型（复用 retriever 已验证的加载路径, 保证离线可用）"""
    global _model_instance, _model_device

    if _model_instance is not None:
        return _model_instance

    # 复用 HybridRetriever 的模型单例: 其 _find_local_model 已覆盖 HF 缓存路径,
    # 且 retriever.py 顶部设置了 HF_HUB_OFFLINE=1, 离线环境不会尝试联网下载
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()
    _model_instance = retriever.model
    _model_device = retriever._device
    return _model_instance


def add_chunks_to_index(
    new_chunks: list,
    document_id: str = None,
) -> int:
    """
    为新增分块生成向量，增量更新 FAISS 索引

    Args:
        new_chunks: [{"chunk_id": str, "content": str, "source": str, ...}, ...]
        document_id: 所属文档 ID (用于后续定向删除)

    Returns:
        成功添加的向量数
    """
    if not new_chunks:
        return 0

    # 1. 加载模型
    model = _get_model()

    # 2. 准备文本和元数据
    texts = [c["content"] for c in new_chunks]
    metadata_list = []
    for c in new_chunks:
        meta = {k: v for k, v in c.items() if k != "content"}
        meta["document_id"] = document_id or meta.get("source", "unknown")
        metadata_list.append(meta)

    # 3. 生成向量
    logger.info(f"正在为 {len(texts)} 个分块生成向量...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings).astype("float32")

    # 4. 写入 Milvus —— 唯一向量后端。
    #    此处是两条 ETL 路径的共同收口, 写在这里保证不会漏。
    #    不可用/写入失败会直接抛错 —— 静默跳过会造成「数据库说成功、索引里没有」。
    from src.vectorstore.milvus_manager import MilvusManager
    milvus = MilvusManager()
    milvus.add_embeddings(embeddings, metadata_list)

    logger.info(f"向量写入完成: Milvus {milvus.total_count} 条")
    return len(embeddings)


def rebuild_vector_index(chunks_file: str = None) -> int:
    """全量重建向量索引 (Milvus 唯一后端)

    从 chunks_processed.jsonl 编码后整批灌入 Milvus —— 先清空 collection 再写。
    不再经过 FAISS 中转。
    """
    if chunks_file is None:
        from src.infra.paths import CHUNKS_PROCESSED_PATH
        chunks_file = CHUNKS_PROCESSED_PATH

    with open(chunks_file, "r", encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]
    logger.info(f"全量向量重建: {len(all_chunks)} 个分块")

    from src.vectorstore.milvus_manager import MilvusManager
    mgr = MilvusManager()
    if not mgr.available:
        raise RuntimeError("Milvus 不可用 —— 无法重建向量索引 (没有降级后端)")
    mgr.reset()   # 全量重建: 先清空 collection

    batch_size = 64
    model = _get_model()
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        embeddings = np.array(model.encode(
            [c["content"] for c in batch],
            normalize_embeddings=True,
            batch_size=batch_size,
        )).astype("float32")

        metadata_list = [
            {
                "chunk_id": c["chunk_id"],
                "source": c.get("source", ""),
                "page": c.get("page", 0),
                "section": c.get("section", ""),
                "document_id": c.get("source", "unknown"),
                # retriever.vector_search 依赖 content 字段 (硬过滤), 必须写入 metadata
                "content": c.get("content", ""),
            }
            for c in batch
        ]
        # add_embeddings 内部已 flush, 逐批落盘
        mgr.add_embeddings(embeddings, metadata_list)
        logger.info(f"  进度: {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    logger.info(f"全量向量重建完成: Milvus {mgr.total_count} 条")
    return len(all_chunks)


if __name__ == "__main__":
    # CLI 模式
    import sys
    if "--rebuild" in sys.argv:
        rebuild_vector_index()
    else:
        print("用法: python src/vector_indexer.py --rebuild")
