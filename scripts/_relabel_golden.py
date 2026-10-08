"""一次性数据修正: Golden QA 意图口径对齐层级式分类 (仅改指定 9 个 ID)。

口径对齐, 不是模型能力提升。改动映射固定且可审计, 其余字段原样保留。
跑完本脚本即完成其历史使命, 不属于线上主流程。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "data", "eval", "finance_qa_golden.jsonl")

# id -> 新意图
RELABEL = {
    "g008": "product_consult",
    "g033": "service_policy",
    "g035": "service_policy",
    "g041": "income_question",
    "g042": "buy_process",
    "g047": "income_question",
    "g058": "service_policy",
    "g101": "buy_process",
    "g104": "complaint",
}

# 改意图后需要同步删除的字段 (g035 不再是 fraud_report, 去掉 risk_required)
DROP_FIELDS = {
    "g035": ["risk_required"],
}

with open(GOLDEN, "r", encoding="utf-8") as f:
    lines = [l for l in f.read().splitlines() if l.strip()]

changed = []
out = []
for line in lines:
    item = json.loads(line)
    gid = item["id"]
    if gid in RELABEL:
        old = item["intent"]
        item["intent"] = RELABEL[gid]
        for field_name in DROP_FIELDS.get(gid, []):
            item.pop(field_name, None)
        changed.append(f"{gid}: {old} -> {item['intent']}")
    out.append(json.dumps(item, ensure_ascii=False))

with open(GOLDEN, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

print(f"共改标 {len(changed)} 条:")
for c in changed:
    print(" ", c)
print(f"总数仍为 {len(out)} 条")
