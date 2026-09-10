"""槽位填充测试 — 金融理财产品槽位 (产品/金额/期限/风险) 提取与追问"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.core.slot_filler import extract_slots, missing_required, followup_question, KNOWN_PRODUCTS


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


def test_missing_required_no_mandatory_in_stage1():
    # 阶段一咨询类无强制槽位; 阶段二资金动账流程再补 product/amount/verification
    assert missing_required("buy_process", {}) == []
    assert missing_required("product_consult", {}) == []


def test_followup_question():
    q = followup_question(["product"])
    assert "哪款产品" in q
    q2 = followup_question(["amount"])
    assert "金额" in q2
