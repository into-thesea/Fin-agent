"""快速通道五道闸: 逐闸正反样本 + AND 关系."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.core.fast_path as fp

CTX = {"local": [{"score": 0.72, "content": "x"}], "sources": ["prod_p002.md"]}
OK = dict(intent="product_consult", contexts=CTX, query="稳盈添利30天的风险等级")


def test_all_gates_pass():
    assert fp.evaluate(**OK) == (True, "pass")


def test_gate1_handoff_blocks():
    got = fp.evaluate(**{**OK, "strong_intent": "fraud_report"})
    assert got == (False, "handoff")


def test_gate1_deposit_insurance_strong_signal_does_not_block():
    # _STRONG_SIGNAL_INTENTS 里有 deposit_insurance, 它是信息型, 必须放行
    assert fp.evaluate(**{**OK, "strong_intent": "deposit_insurance"})[0] is True


def test_gate2_non_informational_blocks():
    assert fp.evaluate(**{**OK, "intent": "buy_process"}) == (False, "intent_not_informational")
    assert fp.evaluate(**{**OK, "intent": "hold_redeem"}) == (False, "intent_not_informational")
    assert fp.evaluate(**{**OK, "intent": "chitchat"}) == (False, "intent_not_informational")


def test_confidence_gate_is_gone():
    """置信度闸实测无区分度, 已被整体移除 (不是留一个恒真的开关) —— 见 spec §4.2"""
    assert not hasattr(fp, "CONFIDENCE_THRESHOLD")
    assert "confidence" not in fp.evaluate.__code__.co_varnames


def test_gate4_no_kb_evidence_blocks():
    assert fp.evaluate(**{**OK, "contexts": {"local": [{"score": 0.31}]}}) == (False, "no_kb_evidence")


def test_empty_retrieval_fails_closed():
    """Milvus 不可用或零命中 → 一律不过闸 → 全部升级到完整管道 (刻意的 fail closed)"""
    assert fp.evaluate(**{**OK, "contexts": {}}) == (False, "no_kb_evidence")
    assert fp.evaluate(**{**OK, "contexts": None}) == (False, "no_kb_evidence")


def test_gate5_private_data_blocks():
    assert fp.evaluate(**{**OK, "query": "我的风险测评等级是多少"}) == (False, "private_data")
    assert fp.evaluate(**{**OK, "query": "帮我查一下持仓"}) == (False, "private_data")


def test_top1_score_handles_missing_shapes():
    assert fp.top1_score(None) == 0.0
    assert fp.top1_score({}) == 0.0
    assert fp.top1_score({"local": []}) == 0.0
    assert fp.top1_score({"local": [{}]}) == 0.0
    assert fp.top1_score(CTX) == 0.72


def test_gates_are_conjunctive():
    """四闸是 AND: 任一条不过, 结果就是不过."""
    assert fp.evaluate(**{**OK, "strong_intent": "complaint", "intent": "chitchat",
                          "contexts": {}, "query": "我的持仓"})[0] is False


def test_route_class_forces_complex_on_handoff():
    assert fp.route_class("你好") == "greeting"
    # classify 收窄后不再有 simple_fact 这一档: 难度判定已整体移交闸门,
    # route_class 只回 greeting|complex (见 plan Pre-flight ruling)
    assert fp.route_class("稳盈添利30天收益多少") == "complex"


def test_route_class_does_not_duplicate_handoff_decision():
    """HANDOFF 判定只该在 evaluate 一处 —— route_class 不得再抄一份"""
    import inspect
    assert "strong_intent" not in inspect.signature(fp.route_class).parameters, \
        "route_class 的 strong_intent 参数在生产中无人传入, 是重复的判定入口"


def test_route_class_forces_complex_on_private_data():
    assert fp.route_class("我的持仓收益怎么样") == "complex"


def test_route_class_keeps_greeting_ahead_of_private_marker():
    # 问候优先: 问候语本身不带私有数据措辞时不受影响
    assert fp.route_class("你好") == "greeting"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK: fast path gates")
