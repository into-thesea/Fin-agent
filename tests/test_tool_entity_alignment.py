"""
理财工具注册表对齐测试 — 产品词表命中 + 工具集边界

核心断言:
  - 产品对齐函数从查询中识别目录产品名 (catalog.jsonl 单一事实源)
  - 只读工具 (query_products/check_suitability/query_holdings) 已注册
  - 资金动账(申购)不注册为 agent 可自由调用工具 → 由 cs_graph 门控节点执行
"""

from src.tools.registry import _query_product_names, EXECUTE_TOOLS, TOOL_DEFINITIONS


def test_query_product_names_matches_catalog():
    names = _query_product_names("稳盈添利30天 收益怎么样")
    assert names and names[0] == "稳盈添利30天"

    # 简称(大额存单)应映射回目录全名(大额存单3年期), 供检索原文对齐
    names2 = _query_product_names("大额存单 有存款保险吗")
    assert names2 and names2[0] == "大额存单3年期"


def test_execute_tools_is_finance_readonly_set():
    assert "retrieve_knowledge" in EXECUTE_TOOLS
    assert {"query_products", "check_suitability", "query_holdings"} <= set(EXECUTE_TOOLS)
    # 资金动账(申购/赎回)不开放给 agent 自由调用
    assert "subscribe_product" not in EXECUTE_TOOLS
    assert "submit_refund" not in EXECUTE_TOOLS


def test_tool_definitions_have_parameters():
    names = [t["name"] for t in TOOL_DEFINITIONS]
    # 4 个核心只读工具必须注册; 账户/交易/卡/贷款/资产/订单等只读查询可增, 不锁死列表
    assert {"retrieve_knowledge", "query_products", "check_suitability", "query_holdings"} <= set(names)
    # 定义与可执行集合同源, 防止只加定义忘了接线 (或反之)
    assert set(names) == set(EXECUTE_TOOLS)
    # 资金动账类不得开放给 agent 自由调用
    assert "subscribe_product" not in names and "submit_refund" not in names
    for t in TOOL_DEFINITIONS:
        assert "parameters" in t and isinstance(t["parameters"], dict)
