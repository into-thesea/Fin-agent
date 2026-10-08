"""
LLMClient function calling 原语单测 (纯函数, 不依赖网络)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.llm import llm_client
from src.llm.llm_client import (
    _parse_deepseek_response,
    append_tool_result,
    build_tools_payload,
    parse_tool_response,
)

TOOLS = [
    {"name": "retrieve_knowledge", "description": "检索知识库",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
]


def test_build_tools_payload_deepseek():
    payload = build_tools_payload(TOOLS, "deepseek")
    assert payload[0]["type"] == "function"
    fn = payload[0]["function"]
    assert fn["name"] == "retrieve_knowledge"
    assert fn["parameters"]["required"] == ["query"]


def test_build_tools_payload_gemini():
    payload = build_tools_payload(TOOLS, "gemini")
    assert hasattr(payload[0], "function_declarations")
    decl = payload[0].function_declarations[0]
    assert decl.name == "retrieve_knowledge"
    assert decl.parameters.type == "OBJECT"


def test_parse_deepseek_tool_calls():
    raw = {
        "choices": [{"message": {
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "retrieve_knowledge", "arguments": '{"query": "存款保险"}'},
            }],
        }}]
    }
    result = _parse_deepseek_response(raw)
    assert result["type"] == "tool_calls"
    call = result["calls"][0]
    assert call["name"] == "retrieve_knowledge"
    assert call["arguments"] == {"query": "存款保险"}


def test_parse_deepseek_tool_calls_bad_json():
    raw = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "function": {"name": "web_search", "arguments": "not-json"}}]}}]}
    result = _parse_deepseek_response(raw)
    assert result["type"] == "tool_calls"
    assert result["calls"][0]["arguments"] == {}


def test_parse_deepseek_text():
    raw = {"choices": [{"message": {"content": "答案文本", "tool_calls": None}}]}
    result = _parse_deepseek_response(raw)
    assert result == {"type": "text", "text": "答案文本"}


def test_parse_deepseek_empty():
    assert _parse_deepseek_response({}) == {"type": "text", "text": ""}


def test_parse_tool_response_provider_dispatch():
    raw_ds = {"choices": [{"message": {"content": "hi"}}]}
    assert parse_tool_response(raw_ds, "deepseek")["text"] == "hi"


def test_append_tool_result_deepseek(monkeypatch):
    monkeypatch.setattr(llm_client, "get_provider", lambda: "deepseek")
    msgs = [{"role": "user", "content": "q"}]
    append_tool_result(msgs, {"id": "call_1", "name": "retrieve_knowledge"}, "结果")
    assert msgs[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "结果"}


def test_append_tool_result_gemini(monkeypatch):
    monkeypatch.setattr(llm_client, "get_provider", lambda: "gemini")
    msgs = [{"role": "user", "content": "q"}]
    append_tool_result(msgs, {"id": "g1", "name": "retrieve_knowledge"}, "结果")
    assert msgs[-1] == {"role": "tool", "name": "retrieve_knowledge", "content": "结果"}


def test_parse_gemini_rest_preserves_thought_signature():
    from src.llm.llm_client import _parse_gemini_rest_response
    raw = {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "retrieve_knowledge", "args": {"query": "存款保险"},
                          "thoughtSignature": "sig_abc"}},
    ]}}]}
    result = _parse_gemini_rest_response(raw)
    assert result["type"] == "tool_calls"
    call = result["calls"][0]
    assert call["name"] == "retrieve_knowledge"
    assert call["arguments"] == {"query": "存款保险"}
    # thoughtSignature 必须保留在原始 dict 中供回传
    assert call["_raw_function_call"]["thoughtSignature"] == "sig_abc"


def test_parse_gemini_rest_text():
    from src.llm.llm_client import _parse_gemini_rest_response
    raw = {"candidates": [{"content": {"parts": [{"text": "答案"}]}}]}
    assert _parse_gemini_rest_response(raw) == {"type": "text", "text": "答案"}


def test_build_gemini_contents_echoes_raw_function_call():
    from src.llm.llm_client import _build_gemini_contents
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "retrieve_knowledge", "arguments": "{}"},
             "_raw_function_call": {"name": "retrieve_knowledge", "args": {"query": "存款保险"},
                                    "thoughtSignature": "sig_abc"}}]},
        {"role": "tool", "name": "retrieve_knowledge", "content": "片段"},
    ]
    contents, system = _build_gemini_contents(msgs)
    assert system == "sys"
    fc = contents[1]["parts"][0]["functionCall"]
    assert fc["thoughtSignature"] == "sig_abc"
    assert fc["args"] == {"query": "存款保险"}
    fr = contents[2]["parts"][0]["functionResponse"]
    assert fr["name"] == "retrieve_knowledge"
    assert fr["response"]["result"] == "片段"


def test_build_tools_payload_rest():
    from src.llm.llm_client import build_tools_payload_rest
    payload = build_tools_payload_rest([{"name": "web_search", "description": "d",
                                         "parameters": {"type": "object", "properties": {}}}])
    assert payload[0]["functionDeclarations"][0]["name"] == "web_search"


def test_to_gemini_tool_messages_roundtrip():
    """OpenAI 风格消息(含 assistant tool_calls + tool) → gemini contents"""
    from src.llm.llm_client import _to_gemini_tool_messages
    msgs = [
        {"role": "system", "content": "你是金融助手"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "retrieve_knowledge", "arguments": {"query": "存款保险"}}}]},
        {"role": "tool", "name": "retrieve_knowledge", "content": "片段"},
    ]
    contents, system = _to_gemini_tool_messages(msgs)
    assert system == "你是金融助手"
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    fc = contents[1]["parts"][0].function_call
    assert fc.name == "retrieve_knowledge"
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0].function_response.name == "retrieve_knowledge"
