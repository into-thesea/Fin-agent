"""仅重建 FAISS 向量索引（基于已扩充的 chunks_processed.jsonl，不覆盖原文件）"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.vector_indexer import rebuild_full_index
from src.infra.paths import CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR

print(f"读取 chunks: {CHUNKS_PROCESSED_PATH}")
print(f"输出索引: {FAISS_INDEX_DIR}")

count = rebuild_full_index(CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR)
print(f"FAISS 索引重建完成，共 {count} 条向量")
