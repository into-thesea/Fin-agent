"""核查意图评测是否存在答案泄漏 (纯本地, 不调 LLM):
对每条 Golden QA, 看 route() 内部的动态 few-shot 检索会不会把它自己/近邻的答案召回。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.agents.router_agent import _retrieve_few_shots

GOLDEN = os.path.join(ROOT, "data", "eval", "finance_qa_golden.jsonl")

rows = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]

self_hit = 0
near_hit = 0
checked = 0
examples = []
for row in rows:
    q = row["question"]
    if row["intent"] == "chitchat":
        continue  # chitchat 不进 few-shot 库
    checked += 1
    shots = _retrieve_few_shots(q, top_k=4)
    shot_qs = [s["question"] for s in shots]
    if q in shot_qs:
        self_hit += 1
        if len(examples) < 6:
            examples.append((row["id"], q, "检索到题目自己"))
    else:
        # 检索到同产品/高度近义
        if any(row["intent"] == s["intent"] for s in shots):
            near_hit += 1

print(f"非chitchat评测题: {checked} 条")
print(f"  动态few-shot直接检索到【题目自己】: {self_hit} 条 ({self_hit/checked*100:.0f}%)")
print(f"  未检索到自己但top4含同意图近邻: {near_hit} 条")
print("\n示例:")
for gid, q, tag in examples:
    print(f"  {gid} [{tag}] {q}")
