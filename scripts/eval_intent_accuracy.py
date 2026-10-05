"""意图准确率在线评测 - 对指定文件每条问题调用RouterAgent，对比标注意图"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.router_agent import RouterAgent

DEFAULT_GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "eval", "finance_qa_golden.jsonl")

parser = argparse.ArgumentParser(description="意图准确率评测")
parser.add_argument("--file", default=DEFAULT_GOLDEN, help="评测集 jsonl 路径")
parser.add_argument("--output", default=None, help="结果输出 json 路径 (默认同目录 intent_accuracy_result.json)")
parser.add_argument("--few-shot", action="store_true", default=False, help="启用动态few-shot (默认关闭)")
args = parser.parse_args()

with open(args.file, "r", encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]

out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.file)),
                                       "intent_accuracy_result.json")

router = RouterAgent()

total = 0
correct = 0
rule_hit = 0
llm_called = 0
latencies = []
per_intent = {}
errors = []

for i, item in enumerate(golden):
    qid = item["id"]
    question = item["question"]
    expected = item["intent"]

    t0 = time.time()
    try:
        result = router.route(question, use_few_shot=args.few_shot)
        latency = time.time() - t0
        latencies.append(latency)
        got = result.intent.value

        # 判断是否走了规则直出
        is_rule = "强信号规则命中" in (result.explanation or "")
        if is_rule:
            rule_hit += 1
        else:
            llm_called += 1

        total += 1
        if got == expected:
            correct += 1
        else:
            errors.append({
                "id": qid,
                "question": question,
                "expected": expected,
                "got": got,
                "confidence": result.confidence,
                "rule": is_rule,
                "explanation": (result.explanation or "")[:120],
            })

        # 按意图统计
        if expected not in per_intent:
            per_intent[expected] = {"total": 0, "correct": 0}
        per_intent[expected]["total"] += 1
        if got == expected:
            per_intent[expected]["correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{len(golden)}, 当前准确率 {correct/total:.1%}, 规则命中 {rule_hit}", flush=True)

    except Exception as e:
        latency = time.time() - t0
        latencies.append(latency)
        errors.append({
            "id": qid,
            "question": question,
            "expected": expected,
            "got": f"ERROR: {type(e).__name__}",
            "error": str(e)[:100],
        })
        total += 1
        print(f"  ERROR {qid}: {e}", flush=True)

print()
print("=" * 60)
print(f"意图准确率评测结果 (共{total}条, 文件={os.path.basename(args.file)})")
print("=" * 60)
print(f"总体准确率: {correct}/{total} = {correct/total:.1%}")
print(f"规则直出: {rule_hit}条 ({rule_hit/total:.1%}), LLM调用: {llm_called}条")
if latencies:
    print(f"平均延迟: {sum(latencies)/len(latencies):.2f}s")
    print(f"延迟P50: {sorted(latencies)[len(latencies)//2]:.2f}s")
    print(f"延迟P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")

print()
print("--- 各意图准确率 ---")
for intent in sorted(per_intent.keys()):
    d = per_intent[intent]
    print(f"  {intent}: {d['correct']}/{d['total']} = {d['correct']/d['total']:.1%}")

print()
print(f"--- 错误案例 ({len(errors)}条) ---")
for e in errors:
    print(f"  {e['id']} [{e['expected']}-> {e['got']}] (conf={e.get('confidence','?')}, rule={e.get('rule','?')})")
    print(f"    Q: {e['question'][:60]}")
    if e.get("explanation"):
        print(f"    理由: {e['explanation']}")

# 保存结果
result = {
    "eval_file": os.path.basename(args.file),
    "total": total,
    "correct": correct,
    "accuracy": correct / total,
    "rule_hit": rule_hit,
    "llm_called": llm_called,
    "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
    "per_intent": per_intent,
    "errors": errors,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {out_path}")
