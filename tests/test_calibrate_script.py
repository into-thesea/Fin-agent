"""标定脚本的扫描逻辑 (不触网)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calibrate_fast_path import sweep

ROWS = [
    # a: 全绿过闸
    dict(intent="product_consult", confidence=0.98, top1=0.80, query="a", strong=None, correct=True),
    # b: 过闸但意图判错 → 应计入 wrong
    dict(intent="product_consult", confidence=0.97, top1=0.75, query="d", strong=None, correct=False),
    # c: 检索证据不足 → 闸④拦
    dict(intent="product_consult", confidence=0.95, top1=0.55, query="b", strong=None, correct=False),
    # d: 动账类 → 闸②拦
    dict(intent="buy_process", confidence=0.99, top1=0.90, query="c", strong=None, correct=True),
]


def test_sweep_reports_coverage_and_errors():
    out = sweep(ROWS, Ts=[0.90], Ss=[0.60])
    assert len(out) == 1
    row = out[0]
    assert row["T"] == 0.90 and row["S"] == 0.60
    assert row["fast"] == 2          # a、d 过闸; b 因 top1<0.60 不过; c 因 buy_process 不过
    assert row["wrong"] == 1         # 过闸的两条里 d 是意图判错的
    assert row["coverage"] == 0.5


def test_sweep_supports_disabled_confidence():
    out = sweep(ROWS, Ts=[None], Ss=[0.60])
    assert out[0]["T"] is None
    assert out[0]["fast"] == 2       # 关掉闸③不改变这个样本集的结果


def test_sweep_restores_thresholds():
    import src.core.fast_path as fp
    saved = (fp.CONFIDENCE_THRESHOLD, fp.KB_SCORE_THRESHOLD)
    sweep(ROWS, Ts=[0.99], Ss=[0.99])
    assert (fp.CONFIDENCE_THRESHOLD, fp.KB_SCORE_THRESHOLD) == saved


if __name__ == "__main__":
    test_sweep_reports_coverage_and_errors()
    test_sweep_supports_disabled_confidence()
    test_sweep_restores_thresholds()
    print("OK: calibrate sweep")
