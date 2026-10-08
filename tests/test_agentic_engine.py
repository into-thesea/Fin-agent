"""
AgenticAgent 工具循环单测 (stub LLM + fake 工具, 不依赖网络/模型)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest

import src.agents.agentic_agent as aa


class StubClient:
    """按脚本顺序返回 chat_with_tools 响应; chat/chat_stream 用于回退"""

    def __init__(self, tool_responses, text_response="最终答案"):
        self.responses = list(tool_responses) + [{"type": "text", "text": text_response}]
        self.chat_calls = []

    def chat_with_tools(self, messages, tools, tool_choice="auto", temperature=0.1):
        return self.responses.pop(0)

    def chat(self, messages, temperature=0.1, json_mode=False):
        self.chat_calls.append(messages)
        return "回退回答"

    def chat_stream(self, messages, temperature=0.1, json_mode=False):
        yield "回退"

    @property
    def provider(self):
        return "gemini"


# 注入 fake 工具, 避免加载 BGE 模型
FAKE_TOOLS = {
    "retrieve_knowledge": lambda query: {
        "text": f"【知识库检索结果】\n- {query}相关片段 [来源: X.pdf]",
        "sources": ["X.pdf"],
        "contexts": {"local": [{"content": f"{query}相关片段", "source": "X.pdf"}],
                     "global": [], "graph": []},
    },
}


def make_agent(script=None, text="最终答案"):
    stub = StubClient(script or [], text_response=text)
    agent = aa.AgenticAgent(system_prompt="你是金融分析师。", intent="fact", client=stub)
    return agent, stub


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    monkeypatch.setattr(aa, "EXECUTE_TOOLS", FAKE_TOOLS)


def test_tool_loop_calls_tool_then_answers():
    script = [{"type": "tool_calls", "calls": [
        {"id": "c1", "name": "retrieve_knowledge", "arguments": {"query": "稳盈添利收益"}}]}]
    agent, stub = make_agent(script)
    result = agent.answer("稳盈添利收益是多少?")
    assert result["answer"] == "最终答案"
    assert result["sources"] == ["X.pdf"]
    assert result["contexts"]["local"]


def test_tool_loop_direct_answer_forces_retrieval():
    """模型不调工具就作答 → 系统补一次检索, 答案与来源都来自真实检索。

    旧断言是 `sources == []`(模型不调工具就放它过) —— 那正是 2026-09-22 线上实测的
    故障: 复杂问题零依据, 而提示词要求"标注来源", 模型就照范例编了个来源名。
    契约已改: 除闲聊/问候外, 一次工具都不调不许直接作答。
    参数说明: 第一轮是模型"不调工具的直接作答", 补检索后需要第二轮文本, 故脚本两条。
    """
    agent, stub = make_agent(script=[{"type": "text", "text": "不调工具的初答"}])
    result = agent.answer("介绍下稳盈添利30天")
    assert result["answer"] == "最终答案"        # 用补检索后的重答
    assert result["sources"] == ["X.pdf"]        # 来源来自真实检索, 不再为空


def test_unknown_tool_does_not_crash():
    script = [{"type": "tool_calls", "calls": [
        {"id": "c1", "name": "not_a_real_tool", "arguments": {}}]}]
    agent, stub = make_agent(script)
    result = agent.answer("q")
    assert result["answer"] == "最终答案"  # 未知工具返回提示后继续循环


def test_fallback_on_tool_loop_error(monkeypatch):
    """工具循环失败 → 回退走 retrieve_knowledge (含实体对齐抑制), 不再用 raw 预取上下文"""
    class FailingClient(StubClient):
        def chat_with_tools(self, messages, tools, tool_choice="auto", temperature=0.1):
            raise RuntimeError("LLM 调用失败")

    # 回退路径统一走 retrieve_knowledge, 这里 mock 它 (确定性返回)
    fake_kr = {
        "text": "【知识库检索结果】\n理财产品相关片段 [来源: X.pdf]",
        "sources": ["X.pdf"],
        "contexts": {"local": [{"content": "理财产品相关片段", "source": "X.pdf"}],
                     "global": [], "graph": []},
    }
    monkeypatch.setattr("src.tools.registry.retrieve_knowledge",
                        lambda query, top_k=5: fake_kr)

    agent = aa.AgenticAgent(system_prompt="你是金融分析师。", intent="fact",
                            client=FailingClient([]))
    result = agent.answer("稳盈添利收益是多少?")
    assert result["answer"] == "回退回答"
    assert result["sources"] == ["X.pdf"]


def test_gather_context_for_stream():
    script = [{"type": "tool_calls", "calls": [
        {"id": "c1", "name": "retrieve_knowledge", "arguments": {"query": "存款保险"}}]}]
    agent, stub = make_agent(script)
    events = []
    gathered = agent._gather_context("稳盈添利收益", on_event=events.append)
    assert "X.pdf" in "".join(gathered["ctx_text"])
    assert gathered["sources"] == ["X.pdf"]
    assert any(e.get("type") == "stage" for e in events)


def test_stream_yields_tokens_then_result():
    script = [{"type": "tool_calls", "calls": [
        {"id": "c1", "name": "retrieve_knowledge", "arguments": {"query": "存款保险"}}]}]
    agent, stub = make_agent(script)
    out = list(agent.answer_stream("稳盈添利收益", on_event=lambda m: None))
    # 最后一个元素是 (result, True)
    token_txt = "".join(t for t, done in out if not done)
    final = out[-1][0]
    assert final["answer"] == token_txt
    assert final["sources"] == ["X.pdf"]
