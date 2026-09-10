"""理财工具注册与 handoff 传播测试"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest
import src.agents.agentic_agent as aa


def test_finance_tools_registered_readonly():
    from src.tools.registry import TOOL_DEFINITIONS, EXECUTE_TOOLS
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert {"retrieve_knowledge", "query_products", "check_suitability",
            "query_holdings"} <= names
    assert "web_search" not in names
    assert "query_graph" not in names
    # 资金动账不注册为 agent 可自由调用工具
    assert "subscribe_product" not in names
    assert {"retrieve_knowledge", "query_products", "check_suitability"} <= set(EXECUTE_TOOLS)


class _StubClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_with_tools(self, messages, tools, tool_choice="auto", temperature=0.1):
        return self.responses.pop(0)

    def chat(self, messages, temperature=0.1, json_mode=False):
        return "回退回答"


def test_agentic_collects_handoff():
    """工具返回 handoff=True (如适当性不匹配) → AgenticAgent 最终结果带 handoff"""
    def fake_suitability(**kwargs):
        return {"text": "测评等级 R2 低于产品 R3, 建议转人工", "sources": [],
                "contexts": {"local": [], "global": [], "graph": []},
                "handoff": True}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(aa, "EXECUTE_TOOLS", {"check_suitability": fake_suitability})
    try:
        script = [{"type": "tool_calls", "calls": [
            {"id": "c1", "name": "check_suitability", "arguments": {"product_id_or_name": "P004"}}]}]
        stub = _StubClient(script + [{"type": "text", "text": "已为您转人工坐席"}] * 2)
        agent = aa.AgenticAgent(system_prompt="你是理财客服助手。", intent="risk_suitability", client=stub)
        result = agent.answer("我测评R2能买平衡增利180天吗")
        assert result["handoff"] is True
        assert result["answer"] == "已为您转人工坐席"
    finally:
        monkeypatch.undo()


def test_agentic_no_handoff_by_default():
    """无 handoff 的工具结果 → 默认 False"""
    def fake_kb(**kwargs):
        return {"text": "知识库片段", "sources": ["X.pdf"],
                "contexts": {"local": [], "global": [], "graph": []}}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(aa, "EXECUTE_TOOLS", {"retrieve_knowledge": fake_kb})
    try:
        script = [{"type": "tool_calls", "calls": [
            {"id": "c1", "name": "retrieve_knowledge", "arguments": {"query": "保本"}}]}]
        stub = _StubClient(script + [{"type": "text", "text": "理财非存款"}] * 2)
        agent = aa.AgenticAgent(system_prompt="你是理财客服助手。", intent="income_question", client=stub)
        result = agent.answer("这个保本吗")
        assert result.get("handoff", False) is False
    finally:
        monkeypatch.undo()
