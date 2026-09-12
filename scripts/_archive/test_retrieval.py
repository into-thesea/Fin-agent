"""检索测试：验证新增知识库内容可被检索到"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.retriever import HybridRetriever

retriever = HybridRetriever()

test_queries = [
    "货币基金和债券基金有什么区别",
    "医疗险和重疾险哪个好",
    "国债和存款的区别",
    "家庭资产怎么配置",
    "R2风险等级是什么意思",
    "增额终身寿和年金险区别",
    "可转债是什么",
    "信托产品门槛",
]

print("=" * 60)
print("检索测试（验证新增知识库内容）")
print("=" * 60)

for q in test_queries:
    print(f"\n【查询】{q}")
    results = retriever.hybrid_retrieve(q, top_k=3)
    local = results.get("local", [])
    for i, r in enumerate(local):
        source = r.get("source", "未知")
        content = r.get("content", "")[:80].replace("\n", " ")
        score = r.get("rrf_score", r.get("score", 0))
        print(f"  [{i+1}] {source} | score={score:.4f}")
        print(f"      {content}...")

print("\n" + "=" * 60)
print("检索测试完成")
print("=" * 60)
