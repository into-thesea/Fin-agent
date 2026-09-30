"""复杂路径（工具循环）必须有真实检索依据 —— 不许"没调工具就作答、来源靠编"。

背景（2026-09-22 实测）: FINANCE_KB_SYSTEM_PROMPT + TOOL_GUIDE 下, qwen-flash 首次调用
直接返回 text、不调任何工具; 而 _answer_with_tools 只发 system+query、不喂上下文, 于是:

  - 答案零依据 (实测复杂问题全部 sources=0)
  - 提示词要求"回答须标注来源（如 [来源: 产品说明书-稳盈添利30天]）", 模型照这个范例
    编出了一个它从未检索过的来源名

覆盖三件事:
  1. 模型不调工具 → 确定性补一次 retrieve_knowledge, 并让模型基于真原文重答
  2. 模型自己调了工具 → 不重复补检索
  3. 提示词里不得再出现"具体来源名"的范例(那是诱发编造的样例)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest

from src.agents import agentic_agent
from src.agents.agentic_agent import AgenticAgent
from src.agents.prompts import FINANCE_KB_SYSTEM_PROMPT


class FakeClient:
    """按脚本返回响应, 并记录每次请求的 messages。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat_with_tools(self, messages, tools, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self.script.pop(0) if self.script else {"type": "text", "text": "（无脚本）"}

    def chat(self, messages, **kwargs):        # 回退路径用
        self.calls.append([dict(m) for m in messages])
        return "回退答案"


@pytest.fixture
def fake_retrieve(monkeypatch):
    """替换 retrieve_knowledge, 记录调用次数, 返回可识别的原文与来源。"""
    seen = {"n": 0}

    def _fake(query, top_k=5):
        seen["n"] += 1
        return {"text": f"【知识库检索结果】\n- 代销产品与自营的关键区别…… [来源: agency_products.md]",
                "sources": ["agency_products.md"],
                "contexts": {"local": [{"content": "代销产品与自营的关键区别", "source": "agency_products.md"}],
                             "global": [], "graph": []}}

    monkeypatch.setitem(agentic_agent.EXECUTE_TOOLS, "retrieve_knowledge", _fake)
    return seen


def _agent(client):
    return AgenticAgent(system_prompt="SYS", intent="product_compare", client=client)


def test_no_tool_call_forces_retrieval(fake_retrieve):
    """模型不调工具就作答 → 补一次检索, 并把原文喂回去重答。"""
    client = FakeClient([
        {"type": "text", "text": "代销产品与自营理财的区别是…… [来源: 产品说明书-编的]"},
        {"type": "text", "text": "代销产品与自营理财的区别是…… [来源: agency_products.md]"},
    ])
    out = _agent(client).answer("代销产品和自营理财有什么区别？")

    assert fake_retrieve["n"] == 1, "未调工具时必须确定性补一次检索"
    assert len(client.calls) == 2, "补检索后要再问一次模型"
    second = "\n".join(m.get("content") or "" for m in client.calls[1])
    assert "agency_products.md" in second, "重答时必须带上真实检索原文"
    assert out["sources"] == ["agency_products.md"], "来源应来自真实检索"
    assert out["answer"].endswith("agency_products.md]"), "最终答案用重答那次"


def test_existing_tool_call_not_double_retrieved(fake_retrieve):
    """模型自己调了工具 → 不重复补检索。"""
    client = FakeClient([
        {"type": "tool_calls", "calls": [{"id": "c1", "name": "retrieve_knowledge",
                                          "arguments": {"query": "代销"}}]},
        {"type": "text", "text": "答案 [来源: agency_products.md]"},
    ])
    out = _agent(client).answer("代销产品和自营理财有什么区别？")

    assert fake_retrieve["n"] == 1, "模型已自行检索, 不应再补"
    assert out["sources"] == ["agency_products.md"]


def test_prompt_has_no_concrete_source_example():
    """提示词不得再用"具体来源名"当范例 —— 模型会把它当模板编出来。"""
    assert "产品说明书-稳盈添利30天" not in FINANCE_KB_SYSTEM_PROMPT
    assert "retrieve_knowledge" in FINANCE_KB_SYSTEM_PROMPT, "须明确要求先调检索工具"
