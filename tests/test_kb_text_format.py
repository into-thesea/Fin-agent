"""KB 文本构造: 产品对齐守卫必须能被单独复用."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.registry import format_kb_text

CTX_HIT = {"local": [{"score": 0.7, "content": "稳盈添利30天 业绩比较基准 3.1%"}]}
CTX_MISS = {"local": [{"score": 0.7, "content": "大额存单 利率 2.1%"}]}


def test_format_returns_text_and_sources():
    out = format_kb_text("随便问问", CTX_HIT)
    assert set(out.keys()) == {"text", "sources"}
    assert "知识库检索结果" in out["text"]


def test_format_empty_contexts_is_fail_closed():
    assert "未检索到" in format_kb_text("随便问问", {})["text"]
    assert "未检索到" in format_kb_text("随便问问", None)["text"]


def test_alignment_guard_drops_content_when_named_product_absent():
    out = format_kb_text("稳盈添利30天怎么样", CTX_MISS)
    assert "未检索到" in out["text"]
    assert out["sources"] == []


def test_alignment_guard_passes_when_product_present():
    out = format_kb_text("稳盈添利30天怎么样", CTX_HIT)
    assert "未检索到" not in out["text"]


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
    print("OK: kb text format")
