"""
BM25 索引重建脚本

用法:
    python scripts/rebuild_bm25.py

在 ETL 流水线中添加新文档后运行，使稀疏检索索引保持同步。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.bm25_index import rebuild_bm25_index
from src.paths import CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH
from src.retrieval.retriever import HybridRetriever


def main():
    print(f"📂 读取分块数据: {CHUNKS_PROCESSED_PATH}")
    if not os.path.exists(CHUNKS_PROCESSED_PATH):
        print("❌ 分块文件不存在，请先运行 ETL 流水线")
        sys.exit(1)

    index = rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)
    print(f"✅ BM25 索引已保存: {BM25_INDEX_PATH}")
    print(f"  文档数: {index.nd}")
    print(f"  词汇量: {len(index.idf)}")

    # 测试检索
    test_queries = ["存款保险", "代销", "风险测评", "赎回", "转人工"]
    print("\n🔍 测试检索:")
    for q in test_queries:
        results = index.search(q, top_k=3)
        if results:
            print(f"  '{q}' → 命中 {len(results)} 条, 最佳: {results[0]['source']} (score={results[0]['score']:.3f})")
        else:
            print(f"  '{q}' → 无结果")

    # 清理旧 retriever 缓存 (使新索引即时生效)
    import gc
    gc.collect()
    print("\n✅ BM25 索引重建完成")


if __name__ == "__main__":
    main()
