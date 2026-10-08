"""临时恢复原标注, 用于和10月4号同口径对比. 不覆盖改标后的文件."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "data", "eval", "finance_qa_golden.jsonl")
OUT = os.path.join(ROOT, "data", "eval", "finance_qa_golden_original.jsonl")

# 改标后的意图 -> 原标注意图 (反向映射)
REVERSE = {
    "g008": "deposit_insurance",
    "g033": "deposit_insurance",
    "g035": "fraud_report",
    "g041": "product_consult",
    "g042": "product_consult",
    "g047": "product_consult",
    "g058": "risk_suitability",
    "g101": "product_consult",
    "g104": "service_policy",
}

with open(GOLDEN, "r", encoding="utf-8") as f:
    lines = [l for l in f.read().splitlines() if l.strip()]

out = []
for line in lines:
    item = json.loads(line)
    gid = item["id"]
    if gid in REVERSE:
        item["intent"] = REVERSE[gid]
        if gid == "g035":
            item["risk_required"] = True
    out.append(json.dumps(item, ensure_ascii=False))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

print(f"原标注版本已保存: {OUT}")
print(f"共 {len(out)} 条, 恢复了 {len(REVERSE)} 条标注")
