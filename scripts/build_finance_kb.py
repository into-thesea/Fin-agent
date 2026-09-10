"""
构建金融理财产品知识库索引 (阶段一/二用)

从 data/finance_kb/*.md 按 '## ' 标题分节生成 chunk (跳过 _ 开头的文件, 如 catalog 由 jsonl 提供),
重建活动索引到 data/output_analysis/ (chunks_processed.jsonl + FAISS + BM25),
若 Milvus 可用则同时把同一批向量灌入 collection (kefu_chunks) — 保证 Milvus/FAISS 两路一致,
避免"向量走电商、稀疏走金融"的混搭。

用法:
    .venv/Scripts/python.exe scripts/build_finance_kb.py [--skip-milvus]
"""

import os
import re
import sys
import json
import glob
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_finance_kb")

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "finance_kb")


def parse_faq_markdown(md_path: str, source: str) -> list:
    """按 '## ' 标题切分为 section chunk (与电商版 parse_faq_markdown 契约一致)"""
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    sections = []
    current_title = None
    current_lines = []
    for line in lines:
        if line.startswith("## "):
            if current_title:
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = []
        elif line.strip() and current_title is not None:
            current_lines.append(line.strip())
    if current_title:
        sections.append((current_title, current_lines))

    stem = os.path.splitext(os.path.basename(md_path))[0]
    chunks = []
    for i, (title, body_lines) in enumerate(sections):
        content = f"【{title}】\n" + "\n".join(body_lines)
        chunks.append({
            "chunk_id": f"kb_{stem}_{i:04d}",
            "source": source,
            "page": 1,
            "content": content,
        })
    logger.info("解析 %s → %d chunk", os.path.basename(md_path), len(chunks))
    return chunks


def collect_chunks() -> list:
    """收集 finance_kb 目录下所有 md 的 chunk (跳过 _ 开头文件)"""
    files = sorted(glob.glob(os.path.join(KB_DIR, "*.md")))
    chunks = []
    for f in files:
        if os.path.basename(f).startswith("_"):
            continue
        chunks.extend(parse_faq_markdown(f, os.path.basename(f)))
    if not chunks:
        logger.error("未收集到任何 finance_kb chunk, 终止")
        sys.exit(1)
    logger.info("finance_kb 共 %d 个 chunk", len(chunks))
    return chunks


def rebuild_local(chunks: list) -> None:
    """写 chunks_processed.jsonl + 重建 FAISS + BM25"""
    from src.infra.paths import (
        CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR, BM25_INDEX_PATH,
    )
    os.makedirs(os.path.dirname(CHUNKS_PROCESSED_PATH), exist_ok=True)
    with open(CHUNKS_PROCESSED_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("chunks 写入: %s (%d)", CHUNKS_PROCESSED_PATH, len(chunks))

    from src.retrieval.vector_indexer import rebuild_full_index
    rebuild_full_index(CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR)

    from src.retrieval.bm25_index import rebuild_bm25_index
    rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)
    logger.info("本地 FAISS+BM25 重建完成")


def sync_milvus() -> None:
    """若 Milvus 可用, 从 FAISS 索引 reconstruct 向量后重灌 collection (保持两路一致)"""
    import faiss
    from src.infra.paths import FAISS_INDEX_DIR
    from src.vectorstore.milvus_manager import MilvusManager

    mgr = MilvusManager()
    if not mgr.available:
        logger.info("Milvus 不可用, 跳过 (系统将走 FAISS 向量后端)")
        return

    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    meta_path = os.path.join(FAISS_INDEX_DIR, "metadata.json")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        logger.error("FAISS 索引文件缺失, 无法同步 Milvus")
        return
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    vectors, metas = [], []
    for i, meta in enumerate(metadata):
        if not meta.get("content"):
            continue
        vectors.append(index.reconstruct(i).reshape(1, -1))
        metas.append({
            "content": meta.get("content", ""),
            "source": meta.get("source", ""),
            "document_id": meta.get("document_id", ""),
            "page": int(meta.get("page", 1)),
            "chunk_id": meta.get("chunk_id", ""),
        })
    if not vectors:
        logger.error("无有效向量可同步 Milvus")
        return
    mgr.reset()
    mgr.add_embeddings(np.vstack(vectors).astype("float32"), metas)
    logger.info("Milvus 已重灌 %d 条 (collection=%s)", mgr.total_count, mgr._collection_name)


def main():
    skip_milvus = "--skip-milvus" in sys.argv
    chunks = collect_chunks()
    rebuild_local(chunks)
    if not skip_milvus:
        sync_milvus()
    logger.info("金融知识库索引构建完成")


if __name__ == "__main__":
    main()
