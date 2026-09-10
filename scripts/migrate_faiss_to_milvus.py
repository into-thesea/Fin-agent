"""
P0: FAISS → Milvus 向量迁移脚本

从 FAISS 索引 (index.faiss + metadata.json) reconstruct 向量，灌入 Milvus collection。
幂等：Milvus 已有同等数量实体时跳过（除非 --force）。

用法:
  .venv/Scripts/python.exe scripts/migrate_faiss_to_milvus.py [--force]

依赖:
  - Milvus 容器已启动 (docker compose up -d etcd minio milvus)
  - pymilvus 已安装
"""

from __future__ import annotations

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("migrate_faiss_to_milvus")

from src.infra.paths import FAISS_INDEX_DIR
from src.vectorstore.milvus_manager import MilvusManager


def load_faiss_data():
    """读取 FAISS 索引与元数据，返回 (vectors, metadata)。过滤 deleted 条目。"""
    import faiss
    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    meta_path = os.path.join(FAISS_INDEX_DIR, "metadata.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        logger.error("FAISS 索引文件缺失: %s", index_path)
        sys.exit(1)

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vectors, metas = [], []
    for i, meta in enumerate(metadata):
        if meta.get("deleted"):
            continue
        if "content" not in meta or not meta.get("content"):
            continue
        vec = index.reconstruct(i).reshape(1, -1)
        vectors.append(vec)
        metas.append({
            "content": meta.get("content", ""),
            "source": meta.get("source", ""),
            "document_id": meta.get("document_id", ""),
            "page": int(meta.get("page", 1)),
            "chunk_id": meta.get("chunk_id", meta.get("source", "")),
        })
    if not vectors:
        logger.error("FAISS 无有效向量可迁移")
        sys.exit(1)
    logger.info("FAISS 读取: %d 条有效向量 (共 %d 条元数据)", len(vectors), len(metadata))
    return np.vstack(vectors).astype("float32"), metas


def verify(mgr: MilvusManager, vectors: np.ndarray, metas: list):
    """用 FAISS 向量本身查询 Milvus，校验 top-1 应命中自己（余弦自相似≈1）。"""
    if len(vectors) < 3:
        logger.warning("向量过少，跳过自检")
        return True
    ok = True
    for i in [0, len(vectors) // 2, len(vectors) - 1]:
        q = vectors[i].reshape(1, -1)
        res = mgr.search(q, top_k=3)
        if not res:
            logger.warning("自检失败: 查询无返回")
            ok = False
            continue
        top = res[0]
        self_chunk = metas[i]["chunk_id"]
        hit_self = (top.get("chunk_id") == self_chunk) or (abs(top.get("score", 0) - 1.0) < 0.05)
        logger.info(
            "自检[%d] chunk=%s 分数=%.4f top1_chunk=%s %s",
            i, self_chunk, top.get("score", 0), top.get("chunk_id"),
            "✓" if hit_self else "✗",
        )
        ok = ok and hit_self
    return ok


def main():
    force = "--force" in sys.argv
    mgr = MilvusManager()

    if not mgr.available:
        logger.error(
            "Milvus 不可用。请先:\n"
            "  1) 安装 pymilvus: .venv/Scripts/python.exe -m pip install pymilvus\n"
            "  2) 启动容器: docker compose -f docker-compose.yml up -d etcd minio milvus"
        )
        sys.exit(1)

    vectors, metas = load_faiss_data()

    # 幂等：已有同等数量且未强制 → 跳过
    if not force and mgr.total_count == len(vectors):
        logger.info("Milvus 已有 %d 条实体, 跳过迁移 (--force 可重建)", mgr.total_count)
    else:
        if force:
            mgr.reset()
        logger.info("开始灌入 %d 条向量...", len(vectors))
        mgr.add_embeddings(vectors, metas)

    # 校验
    ok = verify(mgr, vectors, metas)
    logger.info("Milvus 实体数: %d (期望 %d)", mgr.total_count, len(vectors))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
