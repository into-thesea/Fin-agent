"""
Fin-Agent 向量索引构建器 (企业版)

支持:
  - 增量构建 FAISS 索引 (通过 FaissIndexManager)
  - 文档级元数据追踪 (document_id, page, source)
  - 线程安全
"""

import os
import json
import logging
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.cache.faiss_manager import FaissIndexManager

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

    # 4. 通过 FaissIndexManager 增量更新 (线程安全)
    manager = FaissIndexManager()
    manager.add_embeddings(embeddings, metadata_list)

    logger.info(f"向量索引增量更新完成: 总数 {manager.total_count}")
    return len(embeddings)


def rebuild_full_index(
    chunks_file: str = None,
    output_dir: str = None,
) -> int:
    """
    全量重建向量索引

    Args:
        chunks_file: JSONL 分块文件路径
        output_dir: 索引输出目录
    """
    if chunks_file is None:
        from src.infra.paths import CHUNKS_PROCESSED_PATH
        chunks_file = CHUNKS_PROCESSED_PATH
    if output_dir is None:
        from src.infra.paths import FAISS_INDEX_DIR
        output_dir = FAISS_INDEX_DIR

    # 1. 读取全部分块
    all_chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            all_chunks.append(json.loads(line))

    logger.info(f"全量索引重建: {len(all_chunks)} 个分块")

    # 2. 重置索引
    manager = FaissIndexManager()

    # 3. 分批编码并添加 (防止 OOM)
    batch_size = 64
    total = 0
    model = _get_model()

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        embeddings = np.array(embeddings).astype("float32")

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

        if i == 0:
            # 首次：创建新索引
            from src.cache.faiss_manager import FaissIndexManager as FIM
            # Hack: 直接设置 manager 的索引
            import faiss
            manager._index = faiss.IndexFlatIP(embeddings.shape[1])
            manager._metadata = []

        manager.add_embeddings(embeddings, metadata_list)
        total += len(batch)
        logger.info(f"  进度: {total}/{len(all_chunks)}")

    logger.info(f"全量索引重建完成: {total} 条向量")
    return total


if __name__ == "__main__":
    # CLI 模式
    import sys
    if "--rebuild" in sys.argv:
        rebuild_full_index()
    else:
        print("用法: python src/vector_indexer.py --rebuild")
