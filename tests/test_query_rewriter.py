"""查询补全器测试 — 理财客服多轮: 近指词指向上一轮产品 → 补全产品名 (规则式, 无 LLM)"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.core.dialog_state import DialogState
from src.llm.query_rewriter import rewrite_query

PROD = "稳盈添利30天"


def _state(product=PROD):
    return DialogState(
        session_id="t",
        history=[{"role": "user", "query": f"{PROD}收益怎么样", "answer": f"{PROD}…"}],
        slots={"product": product} if product else {},
    )


def test_pronoun_followup_replaces_focal_product():
    """'它/这款/这个' 指向上一轮产品 → 补全为产品名"""
    for q in ["它保本吗", "这款能买吗", "这个收益呢", "这个产品门槛多少", "它们什么风险"]:
        assert PROD in rewrite_query(q, _state()), f"{q} 未补全产品名"


def test_keeps_own_product_name():
    """已含产品名 → 不重复注入、原样"""
    q = "安鑫纯债基金保本吗"
    assert rewrite_query(q, _state()) == q


def test_no_focal_product_not_rewritten():
    """无焦点产品 → 原样(不编造)"""
    assert rewrite_query("它保本吗", _state(product=None)) == "它保本吗"


def test_no_reference_token_not_rewritten():
    """泛指/无关/问候短句 → 原样, 不猜产品"""
    for q in ["谢谢", "再买5万", "那赎回呢", "帮我推荐低风险的"]:
        assert rewrite_query(q, _state()) == q, f"{q} 不应被改写"


def test_no_history_not_rewritten():
    """首轮无需补全"""
    st = _state()
    st.history = []
    assert rewrite_query("它保本吗", st) == "它保本吗"


if __name__ == "__main__":
    test_pronoun_followup_replaces_focal_product()
    test_keeps_own_product_name()
    test_no_focal_product_not_rewritten()
    test_no_reference_token_not_rewritten()
    test_no_history_not_rewritten()
    print("OK: finance rewrite semantics pass")
