import json

with open(r"D:\aiproject\data\eval\same_oct4_sessions_result.json", encoding="utf-8") as f:
    data = json.load(f)

latencies = [r["latency"] for r in data["rows"]]
latencies.sort()
n = len(latencies)
print(f"轮次数: {n}")
print(f"平均延迟: {sum(latencies)/n:.2f}s")
print(f"P50: {latencies[n//2]:.2f}s")
print(f"P95: {latencies[int(n*0.95)]:.2f}s")
print(f"最小: {min(latencies):.2f}s")
print(f"最大: {max(latencies):.2f}s")
print(f"多轮意图准确率: {data['intent_accuracy']:.1%}")
print(f"续轮准确率: {data['followup_intent_accuracy']:.1%} ({data['followup_turns']}轮)")
print(f"合规通过率: {data['compliance_pass_rate']:.1%}")
print(f"指代消解: {data['coreference_resolved_rate']:.1%} ({data['coreference_cases']}例)")
