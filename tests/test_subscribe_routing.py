"""申购门控触发语义: "想申购某产品" 先进产品分析, 只有构成明确下单/补码才进门控."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.router_agent import QueryIntent as I
from src.core.slot_filler import missing_required
from src.graph.cs_graph import _should_enter_subscribe

BUY = I.BUY_PROCESS
CONS = I.PRODUCT_CONSULT
CHAT = I.CHITCHAT


def _s(**kw):
    return {k: v for k, v in kw.items() if v is not None}


def test_subscribe_enter_table():
    cases = [
        # (intent, slots, query, expect, 说明)
        # —— 只有"想申购某产品"(有产品无金额) → 产品分析, 不进门控 ——
        (BUY,  _s(product="稳盈添利30天"), "",          False, "想申购没金额→分析"),
        (BUY,  _s(product="稳盈添利30天"), "我想申购这个产品", False, "带话同样→分析"),
        (CONS, _s(product="稳盈添利30天"), "这个保本吗", False, "纯咨询→分析"),
        (CONS, _s(product="稳盈添利30天", amount=5.0), "5万放里面收益多少", False, "咨询顺带金额仍按分析"),
        # —— 明确产品+金额 → 构成下单指令, 进门控(仍有验证码二次把关) ——
        (BUY,  _s(product="稳盈添利30天", amount=5.0), "申购5万", True, "申购+金额→门控"),
        (BUY,  _s(product="稳盈添利30天", amount=5.0), "确认购买", True, "确认+金额→门控"),
        # —— 续轮补验证码: 命中6位码且已有目标产品 → 门控 ——
        (CONS, _s(product="稳盈添利30天"), "验证码 123456", True, "补码轮→门控"),
        (BUY,  _s(product="稳盈添利30天", amount=5.0), "123456", True, "金额+补码→门控"),
        # —— 无产品 / 无关意图 → 不进 ——
        (BUY,  _s(amount=5.0), "申购5万", False, "没产品无从执行→分析"),
        (CHAT, {}, "你好", False, "闲聊不进"),
    ]
    for intent, slots, q, expect, why in cases:
        got = _should_enter_subscribe(intent, slots, q)
        assert got is expect, f"{why}: intent={intent.value} slots={slots!r} query={q!r} → {got}, want {expect}"


def test_order_predicate_agrees_with_followup():
    """下单判据只有一处定义: 构成下单指令时, 要么进门控(有产品), 要么追问(缺产品), 不会两头落空."""
    for slots, q in [(_s(amount=5.0), "申购5万"), (_s(product="稳盈添利30天", amount=5.0), "申购5万")]:
        enters = _should_enter_subscribe(BUY, slots, q)
        asks = bool(missing_required(BUY.value, slots))
        assert enters or asks, f"下单指令无人处理: slots={slots!r}"
        assert not (enters and asks), f"既进门控又追问: slots={slots!r}"
    # 非下单指令: 既不进门控也不追问
    assert not missing_required(CONS.value, _s(product="稳盈添利30天", amount=5.0))


if __name__ == "__main__":
    test_subscribe_enter_table()
    test_order_predicate_agrees_with_followup()
    print("OK: subscribe routing semantics pass")
