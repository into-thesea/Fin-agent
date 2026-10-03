"""快速通道五道闸: 逐闸正反样本 + AND 关系."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.core.fast_path as fp

CTX = {"local": [{"score": 0.72, "content": "x"}], "sources": ["prod_p002.md"]}
OK = dict(intent="product_consult", confidence=0.98, contexts=CTX, query="稳盈添利30天的风险等级")


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


def test_gate3_low_confidence_blocks():
    assert fp.evaluate(**{**OK, "confidence": 0.50}) == (False, "low_confidence")


def test_gate3_can_be_disabled():
    saved = fp.CONFIDENCE_THRESHOLD
    fp.CONFIDENCE_THRESHOLD = None
    try:
        assert fp.evaluate(**{**OK, "confidence": 0.10})[0] is True
    finally:
        fp.CONFIDENCE_THRESHOLD = saved


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
    """五闸是 AND: 任一条不过, 结果就是不过."""
    assert fp.evaluate(**{**OK, "strong_intent": "complaint", "intent": "chitchat",
                          "confidence": 0.1, "contexts": {},
                          "query": "我的持仓"})[0] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK: fast path gates")
