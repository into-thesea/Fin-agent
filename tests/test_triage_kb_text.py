"""triage 产出的 kb_text 必须与 retrieve_knowledge 同源 (含对齐守卫)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.cs_graph import CsState, _empty_state


def test_state_has_kb_text_field():
    assert "kb_text" in CsState.__annotations__
    assert "strong_intent" in CsState.__annotations__
    assert "fast_path" in CsState.__annotations__


def test_empty_state_initialises_new_fields():
    st = _empty_state("稳盈添利30天怎么样")
    assert st.get("kb_text") == ""
    assert st.get("strong_intent") is None
    assert st.get("fast_path") is False


def test_merged_contexts_format_uses_same_guard():
    from src.tools.registry import format_kb_text
    merged = {"local": [{"score": 0.7, "content": "稳盈添利30天 业绩比较基准 3.1%"}]}
    out = format_kb_text("稳盈添利30天怎么样", merged)
    assert "知识库检索结果" in out["text"]


def test_multi_query_merged_path_is_fail_closed_too():
    """多问题拆分走的是合并分支, 同样必须经过产品对齐守卫 (否则是个绕行口子)"""
    from src.tools.registry import format_kb_text
    merged = {"local": [{"score": 0.9, "content": "货币市场基金 七日年化收益率"}]}
    out = format_kb_text("稳盈添利30天和大额存单哪个好", merged)
    assert out["sources"] == []
    assert "未检索到" in out["text"]


def test_router_exposes_public_strong_signal():
    from src.agents.router_agent import RouterAgent
    assert hasattr(RouterAgent, "strong_signal_rule"), "强信号方法需公开, cs_graph 要直接用它喂闸①"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
    print("OK: triage kb_text")
