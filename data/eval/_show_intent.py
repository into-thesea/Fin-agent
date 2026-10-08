import json
r = json.load(open('data/eval/intent_accuracy_result.json', encoding='utf-8'))
print(f"总体: {r['correct']}/{r['total']} = {r['accuracy']:.1%}")
print(f"规则直出: {r['rule_hit']}, LLM调用: {r['llm_called']}")
print()
print("各意图:")
for intent, d in sorted(r['per_intent'].items()):
    print(f"  {intent}: {d['correct']}/{d['total']} = {d['correct']/d['total']:.1%}")
print()
print(f"错误数: {len(r['errors'])}")
for e in r['errors']:
    print(f"  {e['id']} [{e['expected']} -> {e['got']}] Q: {e['question'][:55]}")
