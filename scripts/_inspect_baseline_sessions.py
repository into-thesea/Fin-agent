import json

with open(r"D:\aiproject\_ab_baseline\data\eval\finance_sessions.jsonl", encoding="utf-8") as f:
    sessions = [json.loads(l) for l in f if l.strip()]

print(f"会话数: {len(sessions)}")
total = sum(len(s["turns"]) for s in sessions)
print(f"总轮次: {total}")
for s in sessions:
    print(f"  {s['session_id']}: {len(s['turns'])}轮 - {s.get('note', '')}")
    for t in s["turns"]:
        print(f"    - {t['query'][:40]} (期望: {t.get('intent', 'N/A')})")
