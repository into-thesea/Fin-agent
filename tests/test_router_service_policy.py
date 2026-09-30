"""服务规则类问题要有自己的意图格 —— 不能被判成闲聊, 也不能没有知识库依据。

背景（2026-09-22 实测）: "哪些情况应该转人工坐席？" 被判成 chitchat → 走闲聊提示词、
不检索 → sources=0。而知识库里 complaints_human.md 有"哪些情况应转人工坐席"整节,
privacy_security.md 有隐私/信息索取整节 —— 是**意图分类法缺了一格**(service_policy),
不是模型偶发失误: 规则兜底里同样没有这一类, 兜底结果也是 UNKNOWN。

覆盖四件事:
  1. 规则兜底能把转人工/隐私类问题分到 service_policy
  2. 闲聊仍只吃问候寒暄(收紧后不许吞业务问题)
  3. 提示词里确实声明了这个意图, 且点明 chitchat 的边界
  4. 它必须落在**知识库依据**的那条路上(finance_node / FINANCE_KB_SYSTEM_PROMPT)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest

from src.agents.router_agent import INTENT_DETECTION_PROMPT, QueryIntent, RouterAgent
from src.graph.cs_graph import INTENT_NODE_MAP


@pytest.mark.parametrize("query", [
    "哪些情况应该转人工坐席？",
    "客服会不会索要我的验证码？",
    "转人工之后多久有人处理？",
])
def test_rule_fallback_classifies_service_policy(query):
    r = RouterAgent._rule_fallback(query)
    assert r.intent == QueryIntent.SERVICE_POLICY, f"{query} 分到了 {r.intent}"


def test_chitchat_still_only_greeting():
    """收紧 chitchat 的定义后, 真问候仍要分到闲聊。"""
    assert RouterAgent._rule_fallback("你好").intent == QueryIntent.CHITCHAT
    assert RouterAgent._rule_fallback("你是谁").intent == QueryIntent.CHITCHAT


def test_prompt_declares_intent_and_bounds_chitchat():
    assert "service_policy" in INTENT_DETECTION_PROMPT
    assert "转人工" in INTENT_DETECTION_PROMPT
    # chitchat 必须写明"仅限问候寒暄", 否则它会继续吞掉业务/政策问题
    idx = INTENT_DETECTION_PROMPT.find("chitchat")
    assert "仅限" in INTENT_DETECTION_PROMPT[idx:idx + 120]


def test_intent_is_kb_grounded():
    """服务规则类必须有知识库依据: 图上落 finance_node(finance KB prompt)。"""
    assert INTENT_NODE_MAP.get(QueryIntent.SERVICE_POLICY) == "finance_node"

    from src.analyst_agent import AnalystAgent  # noqa: F401  (确认可导入)
    from src.agents.prompts import FINANCE_KB_SYSTEM_PROMPT
    src = open(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "src", "analyst_agent.py"), encoding="utf-8").read()
    assert "QueryIntent.SERVICE_POLICY: FINANCE_KB_SYSTEM_PROMPT" in src, \
        "两条 prompt_map 都要把它指向知识库 prompt (analyze / analyze_stream 两条路)"
    assert src.count("QueryIntent.SERVICE_POLICY: FINANCE_KB_SYSTEM_PROMPT") == 2
    assert FINANCE_KB_SYSTEM_PROMPT  # 非空
