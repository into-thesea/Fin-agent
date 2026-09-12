"""验证扩充后的知识库内容"""
import json

print("=== 产品目录验证 ===")
with open("data/finance_kb/catalog.jsonl", "r", encoding="utf-8") as f:
    products = [json.loads(line) for line in f if line.strip()]
print(f"产品总数: {len(products)}")
for p in products:
    pid = p["id"]
    name = p["name"]
    ptype = p["type"]
    risk = p["risk"]
    print(f"  {pid} | {name} | {ptype} | {risk}")

print()
print("=== 知识问答验证 ===")
with open("data/output_analysis/chunks_processed.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]
print(f"chunks总数: {len(chunks)}")
print("新增的13条chunks:")
for c in chunks[-13:]:
    cid = c["chunk_id"]
    src = c["source"]
    preview = c["content"][:60].replace("\n", " ")
    print(f"  {cid} | {src} | {preview}...")

print()
print("=== 产品门类统计 ===")
from collections import Counter
type_counter = Counter(p["type"].split("(")[0] for p in products)
for t, cnt in type_counter.most_common():
    print(f"  {t}: {cnt}款")
