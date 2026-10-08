"""简单事实类问题统一进图, chat.py 不再内联处理."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _chat_src():
    return open(os.path.join(ROOT, "src", "api", "routes", "chat.py"), encoding="utf-8").read()


def test_chat_py_has_no_simple_fact_branch():
    src = _chat_src()
    assert 'qclass in ("greeting", "simple_fact")' not in src
    assert "prompt_map" not in src, "chat.py 仍有内联 prompt 分支"


def test_chat_py_still_handles_greeting_inline():
    """greeting 不查 KB、不需意图, 保留内联是刻意的 (见 spec §3.4)"""
    assert 'if qclass == "greeting"' in _chat_src()


def test_chat_py_does_not_retrieve_inline():
    """内联路径不得再自己检索 —— 检索与判定统一在图里"""
    assert "retrieve_knowledge" not in _chat_src()


if __name__ == "__main__":
    test_chat_py_has_no_simple_fact_branch()
    test_chat_py_still_handles_greeting_inline()
    test_chat_py_does_not_retrieve_inline()
    print("OK: no inline simple_fact")
