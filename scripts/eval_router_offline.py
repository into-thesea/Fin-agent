"""离线评测: 规则混合仲裁在 Golden QA 上的强信号意图准确率"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.router_agent import RouterAgent

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "eval", "finance_qa_golden.jsonl")

with open(GOLDEN, "r", encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]

router = RouterAgent()

# 测试强信号规则仲裁
strong_signal_intents = {"fraud_report", "complaint", "deposit_insurance"}
total = 0
correct = 0
details = []

for item in golden:
    intent = item["intent"]
    if intent not in strong_signal_intents:
        continue
    total += 1
    result, hit_intent = router._strong_signal_rule(item["question"])
    if result is not None and result.intent.value == intent:
        correct += 1
        details.append(f"  OK   {item['id']} [{intent}] {item['question'][:40]}")
    else:
        got = result.intent.value if result else "NO_MATCH"
        details.append(f"  FAIL {item['id']} [{intent}->got:{got}] {item['question'][:40]}")

print(f"=== 强信号规则仲裁评测 ===")
print(f"强信号意图case总数: {total}")
print(f"规则直出正确: {correct}")
print(f"规则直出准确率: {correct/total:.1%}" if total > 0 else "无case")
print()
print("--- 明细 ---")
for d in details:
    print(d)

# 统计各意图分布
print()
print(f"=== Golden QA 意图分布 (共{len(golden)}条) ===")
from collections import Counter
intent_counts = Counter(item["intent"] for item in golden)
for intent, count in intent_counts.most_common():
    print(f"  {intent}: {count}")
