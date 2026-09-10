"""
构建客服知识库索引 (阶段一用)

从 Markdown FAQ 分节生成 chunk，重建 FAISS + BM25 索引。
客服 FAQ 每条/每节即一个 chunk，无需 PDF 分块。

用法:
    python scripts/build_customer_kb.py data/customer_faq/客服FAQ.md
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_faq_markdown(md_path: str, source: str) -> list:
    """按 '## ' 标题切分为 section chunk"""
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    sections = []  # (title, body_lines)
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

    chunks = []
    for i, (title, body_lines) in enumerate(sections):
        content = f"【{title}】\n" + "\n".join(body_lines)
        chunks.append({
            "chunk_id": f"kb_{os.path.splitext(os.path.basename(md_path))[0]}_{i:04d}",
            "source": source,
            "page": 1,
            "content": content,
        })
    logger.info("解析 %d 个 FAQ 分节 → %d 个 chunk", len(sections), len(chunks))
    return chunks


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/build_customer_kb.py <faq.md>")
        sys.exit(1)

    md_path = os.path.abspath(sys.argv[1])
    source = os.path.basename(md_path)

    from src.infra.paths import (
        CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH, FAISS_INDEX_DIR,
    )

    chunks = parse_faq_markdown(md_path, source)

    # 1. 写全量 chunk 文件 (覆盖, 全新客服库)
    os.makedirs(os.path.dirname(CHUNKS_PROCESSED_PATH), exist_ok=True)
    with open(CHUNKS_PROCESSED_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("chunks 写入: %s (%d)", CHUNKS_PROCESSED_PATH, len(chunks))

    # 2. 重建 FAISS 向量索引
    from src.retrieval.vector_indexer import rebuild_full_index
    rebuild_full_index(CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR)

    # 3. 重建 BM25 稀疏索引
    from src.retrieval.bm25_index import rebuild_bm25_index
    rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)

    logger.info("客服知识库构建完成")


if __name__ == "__main__":
    main()
