"""
边界块 / 规则路由单测 (金融理财产品客服版)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.agents.prompts import (
    BOUNDARY_BLOCK,
    BOUNDARY_BLOCK_LIGHT,
    COMPARE_SYSTEM_PROMPT,
    CS_CHITCHAT_SYSTEM_PROMPT,
    FACT_SYSTEM_PROMPT,
    FAQ_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    ORDER_SYSTEM_PROMPT,
    REFUND_SYSTEM_PROMPT,
    TOOL_GUIDE,
    TREND_SYSTEM_PROMPT,
    UNKNOWN_SYSTEM_PROMPT,
)
from src.agents.router_agent import QueryIntent, RouterAgent
from src.llm.query_router import classify


def test_boundary_block_present_in_all_domain_prompts():
    for name, p in [
        ("FACT", FACT_SYSTEM_PROMPT), ("TREND", TREND_SYSTEM_PROMPT),
        ("COMPARE", COMPARE_SYSTEM_PROMPT), ("GENERAL", GENERAL_SYSTEM_PROMPT),
        ("FAQ", FAQ_SYSTEM_PROMPT), ("ORDER", ORDER_SYSTEM_PROMPT),
        ("REFUND", REFUND_SYSTEM_PROMPT),
    ]:
        assert "领域边界" in p, f"{name} 缺领域边界"
        assert "反幻觉" in p, f"{name} 缺反幻觉边界"
        assert "数据版本" in p, f"{name} 缺数据版本边界"
    # 金融合规: 非存款提示语在关键边界中
    assert "非存款" in BOUNDARY_BLOCK
    assert "收益承诺" in BOUNDARY_BLOCK


def test_boundary_light_and_unknown_prompt():
    assert "领域边界" in BOUNDARY_BLOCK_LIGHT
    assert "领域边界" in UNKNOWN_SYSTEM_PROMPT
    assert "领域边界" in CS_CHITCHAT_SYSTEM_PROMPT


def test_tool_guide_mentions_only_kb_tool():
    assert "retrieve_knowledge" in TOOL_GUIDE
    assert "query_products" in TOOL_GUIDE
    assert "check_suitability" in TOOL_GUIDE
    assert "web_search" not in TOOL_GUIDE
    assert "query_graph" not in TOOL_GUIDE
    # 动账不开放给 agent 直接执行
    assert "subscribe_product(" not in TOOL_GUIDE


def test_rule_fallback_returns_unknown_for_meta():
    result = RouterAgent._rule_fallback("你是什么模型？")
    assert result.intent == QueryIntent.UNKNOWN
    assert result.confidence < 0.6


def test_rule_fallback_returns_unknown_for_non_customer():
    result = RouterAgent._rule_fallback("帮我写一段 python 代码")
    assert result.intent == QueryIntent.UNKNOWN


def test_rule_fallback_finance_intents():
    assert RouterAgent._rule_fallback("稳盈添利30天保本吗").intent == QueryIntent.INCOME_QUESTION
    assert RouterAgent._rule_fallback("银行理财是存款吗").intent == QueryIntent.DEPOSIT_INSURANCE
    assert RouterAgent._rule_fallback("我测评R2能买平衡增利180天吗").intent == QueryIntent.RISK_SUITABILITY
    assert RouterAgent._rule_fallback("大额存单和固收理财哪个好").intent == QueryIntent.PRODUCT_COMPARE
    assert RouterAgent._rule_fallback("这个怎么买").intent == QueryIntent.BUY_PROCESS
    assert RouterAgent._rule_fallback("没到期能取出来吗").intent == QueryIntent.HOLD_REDEEM
    assert RouterAgent._rule_fallback("赎回要手续费吗").intent == QueryIntent.FEE_RULE
    assert RouterAgent._rule_fallback("朋友推荐的内部高收益理财靠谱吗").intent == QueryIntent.FRAUD_REPORT
    assert RouterAgent._rule_fallback("我要投诉").intent == QueryIntent.COMPLAINT
    assert RouterAgent._rule_fallback("你好呀").intent == QueryIntent.CHITCHAT


def test_classify_basic_routes():
    # v4.0: classify 只分 greeting/complex —— "简不简单"改由五道闸判定
    # (src/core/fast_path.py), 不再由关键词表决定。原 simple_fact 断言随之作废。
    assert classify("你好") == "greeting"
    assert classify("收益怎么算") == "complex"
    assert classify("存款保险保多少") == "complex"
    assert classify("为什么理财会亏损") == "complex"
    assert classify("我的持仓收益如何") == "complex"
