"""标定脚本的扫描逻辑 (不触网)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.calibrate_fast_path import sweep

ROWS = [
    # a: 全绿过闸
    dict(intent="product_consult", top1=0.80, query="a", strong=None, correct=True),
    # b: 过闸但意图判错 → 应计入 wrong
    dict(intent="product_consult", top1=0.75, query="d", strong=None, correct=False),
    # c: 检索证据不足 → 闸④拦
    dict(intent="product_consult", top1=0.55, query="b", strong=None, correct=False),
    # d: 动账类 → 闸②拦
    dict(intent="buy_process", top1=0.90, query="c", strong=None, correct=True),
]


def test_sweep_reports_coverage_and_errors():
    out = sweep(ROWS, Ss=[0.60])
    assert len(out) == 1
    row = out[0]
    assert row["S"] == 0.60
    assert row["fast"] == 2          # a、d 过闸; b 因 top1<0.60 不过; c 因 buy_process 不过
    assert row["wrong"] == 1         # 过闸的两条里 d 是意图判错的
    assert row["coverage"] == 0.5


def test_sweep_covers_multiple_thresholds():
    out = sweep(ROWS, Ss=[0.50, 0.60, 0.80])
    assert [r["S"] for r in out] == [0.50, 0.60, 0.80]
    # S 越低放行越多
    assert out[0]["fast"] >= out[1]["fast"] >= out[2]["fast"]


def test_sweep_restores_threshold():
    import src.core.fast_path as fp
    saved = fp.KB_SCORE_THRESHOLD
    sweep(ROWS, Ss=[0.99])
    assert fp.KB_SCORE_THRESHOLD == saved


if __name__ == "__main__":
    test_sweep_reports_coverage_and_errors()
    test_sweep_covers_multiple_thresholds()
    test_sweep_restores_threshold()
    print("OK: calibrate sweep")
