"""classify 收窄为只管 greeting; 难度判定已移交五道闸."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.llm.query_router as qr


def test_tables_removed():
    assert not hasattr(qr, "_FAQ_MARKERS")
    assert not hasattr(qr, "_PRODUCT_NAMES")
    assert not hasattr(qr, "_COMPLEX_MARKERS")


def test_greetings_kept():
    assert qr.classify("你好") == "greeting"
    assert qr.classify("谢谢") == "greeting"


def test_classify_never_returns_simple_fact():
    for q in ("稳盈添利30天收益多少", "存款保险赔多少", "提前赎回会收费吗",
              "你好", "帮我申购5万稳盈添利30天"):
        assert qr.classify(q) in ("greeting", "complex"), q


def test_business_request_is_complex():
    assert qr.classify("我要赎回我的理财") == "complex"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
    print("OK: query router narrowed")
