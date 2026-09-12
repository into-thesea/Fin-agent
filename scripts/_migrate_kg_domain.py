"""
一次性迁移: 拆分 kg_triples.jsonl → 领域知识(手写) + 产品三元组(派生)

原 kg_triples.jsonl 里混了两类:
  - 产品三元组: 每产品固定 3 条 (is_type_of / risk_level / deposit_insured),
    全是 catalog.jsonl 字段的机械展开 —— 跟 prod_pNNN.md 的「产品要素」一样,
    手写就是重复源, 新增产品必然漏 (P012-P024 就是漏的)
  - 领域知识: covered_by / not_covered_by / requires / must_not 这类多跳边,
    承载"理财≠存款""存款保险不保理财"等推理, 必须手写

迁移后:
  kg_domain_triples.jsonl  = 手写领域知识 (本脚本产出)
  kg_triples.jsonl         = build_finance_kb 生成 = 领域 + 产品派生

幂等: 目标文件已存在则跳过 (加 --force 覆盖)。
用法: .venv/Scripts/python.exe scripts/_migrate_kg_domain.py [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "data", "finance_kb")
SRC = os.path.join(KB_DIR, "kg_triples.jsonl")
DST = os.path.join(KB_DIR, "kg_domain_triples.jsonl")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 这三个 relation 由 catalog.jsonl 派生, 不再手写
DERIVED_RELATIONS = {"is_type_of", "risk_level", "deposit_insured"}

# ── 新门类需要的领域知识 (债券/信托/贵金属 此前完全没建图) ──
NEW_DOMAIN = [
    # 债券
    ("债券", "is_not", "存款", "bond_knowledge.md", "债券非存款"),
    ("债券", "not_covered_by", "存款保险", "bond_knowledge.md", "债券不受存款保险保障"),
    ("储蓄国债", "is_subtype_of", "债券", "bond_knowledge.md", "国债按发行方式分类"),
    ("记账式国债", "is_subtype_of", "债券", "bond_knowledge.md", "国债按发行方式分类"),
    ("可转债", "is_subtype_of", "债券", "bond_knowledge.md", "可转债属债券"),
    ("储蓄国债", "covered_by", "国家信用", "bond_knowledge.md", "以国家信用为担保, 非存款保险"),
    ("储蓄国债", "allows", "提前兑取分档计息", "bond_knowledge.md", "可提前兑取但损失利息"),
    ("记账式国债", "has_risk", "价格随市场利率波动", "bond_knowledge.md", "可交易故有价格风险"),
    ("可转债", "has_risk", "价格随正股波动", "bond_knowledge.md", "兼具股性"),
    ("可转债", "subject_to", "强制赎回条款", "bond_knowledge.md", "未及时转股可能被低价赎回"),
    ("国债利息", "is", "免征个人所得税", "bond_knowledge.md", "税收优惠"),
    # 信托
    ("信托", "is_not", "存款", "trust_knowledge.md", "信托非存款"),
    ("信托", "not_covered_by", "存款保险", "trust_knowledge.md", "信托不受存款保险保障"),
    ("信托", "requires", "合格投资者认证", "trust_knowledge.md", "金融资产≥300万或近3年年均收入≥50万"),
    ("信托", "must_include", "合格投资者门槛说明", "trust_knowledge.md", "不满足条件不得销售"),
    ("信托", "has_risk", "信用风险", "trust_knowledge.md", "融资方违约风险"),
    ("信托", "has_risk", "流动性风险", "trust_knowledge.md", "封闭期内不可赎回"),
    # 贵金属
    ("账户黄金", "is_not", "存款", "gold_knowledge.md", "账户黄金非存款"),
    ("账户黄金", "not_covered_by", "存款保险", "gold_knowledge.md", "贵金属不受存款保险保障"),
    ("账户黄金", "feature", "不产生利息或现金流", "gold_knowledge.md", "收益仅来自价差"),
    ("账户黄金", "has_risk", "价格随国际金价波动", "gold_knowledge.md", "无固定收益"),
    # 基金细分 (新增的 4 类)
    ("货币市场基金", "is_subtype_of", "公募基金", "fund_knowledge.md", "按投资对象分类"),
    ("指数基金", "is_subtype_of", "公募基金", "fund_knowledge.md", "按投资对象分类"),
    ("QDII基金", "is_subtype_of", "公募基金", "fund_knowledge.md", "按投资对象分类"),
    ("FOF基金", "is_subtype_of", "公募基金", "fund_knowledge.md", "按投资对象分类"),
    ("货币市场基金", "is", "低风险高流动性", "fund_knowledge.md", "风险最低 R1"),
    ("FOF基金", "has_risk", "双重收费", "fund_knowledge.md", "底层基金另收费用"),
    ("QDII基金", "has_risk", "汇率风险", "fund_knowledge.md", "人民币升值侵蚀收益"),
    # 保险细分 (新增的 4 类)
    ("重疾险", "is_subtype_of", "保险产品", "insurance_knowledge.md", "保障型"),
    ("医疗险", "is_subtype_of", "保险产品", "insurance_knowledge.md", "保障型"),
    ("意外险", "is_subtype_of", "保险产品", "insurance_knowledge.md", "保障型"),
    ("定期寿险", "is_subtype_of", "保险产品", "insurance_knowledge.md", "保障型"),
    ("重疾险", "is", "给付型保险", "insurance_knowledge.md", "确诊即给付保额"),
    ("医疗险", "is", "报销型保险", "insurance_knowledge.md", "实报实销有免赔额"),
    ("医疗险", "must_include", "不保证续保提示", "insurance_knowledge.md", "停售或健康变化可能无法续保"),
    ("定期寿险", "feature", "期满无理赔保费不返还", "insurance_knowledge.md", "纯保障型无现金价值"),
    ("意外险", "excludes", "猝死", "insurance_knowledge.md", "猝死通常不属意外, 除非条款特别约定"),
    # 资产配置
    ("家庭资产配置", "requires", "先保障后理财", "asset_allocation.md", "标普四账户原则"),
    ("家庭资产配置", "must_not", "把鸡蛋放在一个篮子里", "asset_allocation.md", "分散投资"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(DST) and not args.force:
        print(f"已存在, 跳过: {os.path.relpath(DST, ROOT)} (--force 覆盖)")
        return 0

    with open(SRC, "r", encoding="utf-8") as f:
        all_triples = [json.loads(line) for line in f if line.strip()]

    domain = [t for t in all_triples if t.get("relation") not in DERIVED_RELATIONS]
    derived = len(all_triples) - len(domain)

    existing = {(t["head"], t["relation"], t["tail"]) for t in domain}
    added = []
    for h, rel, tail, doc, note in NEW_DOMAIN:
        if (h, rel, tail) not in existing:
            added.append({"head": h, "relation": rel, "tail": tail, "doc": doc, "note": note})
            existing.add((h, rel, tail))

    print(f"原 {len(all_triples)} 条 → 领域 {len(domain)} 条 + 派生 {derived} 条(交给 build 生成)")
    print(f"新增门类领域知识 {len(added)} 条 (债券/信托/贵金属/基金细分/保险细分/资产配置)")
    print(f"迁移后领域知识合计: {len(domain) + len(added)} 条")

    if args.dry_run:
        print("\n--dry-run, 未写入")
        return 0

    out = domain + added
    with open(DST, "w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    bk = os.path.join(KB_DIR, f"kg_triples.bak_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    shutil.copy2(SRC, bk)
    print(f"\n已写 {os.path.relpath(DST, ROOT)} ({len(out)} 条)")
    print(f"已备份原文件 → {os.path.relpath(bk, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
