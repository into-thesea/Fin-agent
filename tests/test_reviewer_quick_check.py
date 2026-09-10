"""
Reviewer quick_check 单测 (来源有效性/有货未找到/未引用来源) — 智能客服版
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest
from src.agents.reviewer_agent import ReviewerAgent


def _context(local_sources=("售后政策.pdf",), n=3):
    local = [{"content": f"第{i}段: 京东快递首重15元，续重每公斤5元。退货需在签收后7天内申请。",
              "source": local_sources[0] if i == 0 else local_sources[i % len(local_sources)]}
             for i in range(n)]
    return {"local": local, "global": [], "graph": []}


def test_pass_when_grounded():
    ctx = _context()
    answer = "京东快递首重运费15元。 [来源: 售后政策.pdf]"
    result = ReviewerAgent().quick_check(answer, ctx)
    assert result.verdict == "pass"


def test_flag_citation_outside_retrieved():
    ctx = _context()
    answer = "京东快递首重运费15元。 [来源: 其他文档.pdf]"
    result = ReviewerAgent().quick_check(answer, ctx)
    assert result.verdict == "flag"
    assert any("来源之外" in c.statement for c in result.claims)


def test_flag_not_found_despite_content():
    ctx = _context(n=4)
    answer = "知识库中未找到该信息，无法回答。"
    result = ReviewerAgent().quick_check(answer, ctx)
    assert result.verdict == "flag"
    assert any("却回答" in c.statement or "不匹配" in c.statement for c in result.claims)


def test_flag_long_answer_without_citation():
    ctx = _context()
    answer = "我们的客服体系覆盖售前咨询、售后支持、退换货处理等多个环节，致力于为用户提供及时、专业、贴心的服务体验。" * 4
    result = ReviewerAgent().quick_check(answer, ctx)
    assert result.verdict == "flag"
    assert any("未引用任何检索来源" in c.statement for c in result.claims)


def test_numeric_check_still_works():
    ctx = _context()
    answer = "京东快递首重运费999元。 [来源: 售后政策.pdf]"
    result = ReviewerAgent().quick_check(answer, ctx)
    assert result.verdict == "flag"
    assert any("数字 999" in c.statement for c in result.claims)
