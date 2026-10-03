"""多问题拆分的零延迟快速路径: 单个句末问号不该触发 LLM.

背景: 原判据是"句中含有 ？/；/还有/另外/以及"，而用户提问几乎都以 ？ 结尾 ——
golden 105 条里 103 条(98%)命中, 其中 98 条只是单个问号。改造后 75% 流量经过
route_retrieve_node 会走这里, 每条白花一次 LLM 往返 (实测快答 P50 +1.4s)。

注意: 不能用"抛异常的假 client"来断言没调用 —— decompose_query 有
`except Exception: return [query]`, 会把异常吞掉让测试假绿。改为记录调用次数。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.llm.llm_client as llm
from src.core.query_decomposer import decompose_query

CALLS = []


class _Recorder:
    def __init__(self, *a, **k):
        CALLS.append(1)

    def chat(self, messages, **kw):
        return '["收益怎么算", "风险等级是多少"]'


def _run(query):
    CALLS.clear()
    saved = llm.create_client
    llm.create_client = _Recorder
    try:
        out = decompose_query(query)
    finally:
        llm.create_client = saved
    return out, len(CALLS)


def test_single_trailing_question_mark_skips_llm():
    q = "稳盈添利30天的风险等级是多少？"
    out, calls = _run(q)
    assert calls == 0, "单个句末问号不该调 LLM"
    assert out == [q]


def test_plain_statement_skips_llm():
    q = "稳盈添利30天的风险等级是多少"
    out, calls = _run(q)
    assert calls == 0
    assert out == [q]


def test_two_question_marks_still_decompose():
    out, calls = _run("银行理财是存款吗？存款保险赔吗？")
    assert calls == 1, "真多问题必须仍然走 LLM 拆分"
    assert out == ["收益怎么算", "风险等级是多少"]


def test_explicit_conjunction_still_decomposes():
    out, calls = _run("我想问收益怎么算，还有风险等级是多少")
    assert calls == 1
    assert out == ["收益怎么算", "风险等级是多少"]


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
    print("OK: decompose fast path")
