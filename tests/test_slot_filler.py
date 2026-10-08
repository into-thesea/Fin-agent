"""槽位填充测试 — 金融理财产品槽位 (产品/金额/期限/风险) 提取与追问"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.core.slot_filler import KNOWN_PRODUCTS, extract_slots, followup_question, missing_required


def test_product_from_catalog():
    assert len(KNOWN_PRODUCTS) >= 9
    s = extract_slots("稳盈添利30天 收益怎么样")
    assert s["product"] == "稳盈添利30天"


def test_product_type_by_keyword():
    s = extract_slots("大额存单 受存款保险保障吗")
    assert s["product_type"] == "存款"
    s2 = extract_slots("我想买一份增额终身寿")
    assert "保险" in s2["product_type"]


def test_amount_normalized_to_wan():
    assert extract_slots("想买50万")["amount"] == 50.0
    assert extract_slots("申购50000元")["amount"] == 5.0
    assert extract_slots("投1万元")["amount"] == 1.0


def test_term_and_risk():
    s = extract_slots("放3个月 能买R3吗")
    assert s["term"] == "3个月"
    assert s["risk_level"] == "R3"


def test_entities_supplement():
    s = extract_slots("这个能买吗", entities=[
        {"name": "平衡增利180天", "type": "Product"},
        {"name": "R2", "type": "RiskLevel"},
    ])
    assert s["product"] == "平衡增利180天"
    assert s["risk_level"] == "R2"


def test_missing_required_only_in_execute_flow():
    # 咨询类一律不追问: 信息型问题本来就不保证带得出产品名
    for intent in ["product_consult", "product_compare", "income_question",
                   "fee_rule", "risk_suitability", "hold_redeem", "deposit_insurance"]:
        assert missing_required(intent, {}) == [], f"{intent} 不该追问"
        assert missing_required(intent, extract_slots("稳盈添利30天收益多少")) == []

    # 有购买意图但没有金额 → 还不构成可执行指令, 不追问
    assert missing_required("buy_process", {}) == []
    assert missing_required("buy_process", {"amount": None}) == []

    # 购买意图 + 金额 + 没点名产品 → 追问产品
    assert missing_required("buy_process", {"amount": 50.0}) == ["product"]
    assert missing_required("buy_process", extract_slots("我想买50万")) == ["product"]

    # 购买意图 + 金额 + 有产品 → 不追问
    assert missing_required("buy_process", {"amount": 50.0, "product": "稳盈添利30天"}) == []


def test_golden_never_triggers_followup():
    """回归护栏: 105 条 Golden 没有一条会被槽位闸拦成追问.

    这条曾经被"按产品域强制 product"的规则打破 (75/105 被反问),
    它保护的是线上意图准确率基线 (88.6%)。
    """
    import json
    from pathlib import Path

    golden = Path(__file__).resolve().parent.parent / "data" / "eval" / "finance_qa_golden.jsonl"
    rows = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 100, "Golden 集缺失或过小"

    triggered = []
    for r in rows:
        slots = extract_slots(r.get("question", ""), r.get("entities") or [])
        if missing_required(r.get("intent", ""), slots):
            triggered.append(r.get("id"))
    assert triggered == [], f"这些 Golden 用例被误判为缺槽位: {triggered}"


def test_followup_question():
    q = followup_question(["product"])
    assert "哪款产品" in q
    q2 = followup_question(["amount"])
    assert "金额" in q2
