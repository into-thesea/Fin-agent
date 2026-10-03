"""
金融理财产品智能客服 工具注册表 — function calling 用

定义 Agent 可调用的工具 (声明 + 执行函数)。工具执行结果统一返回 dict:
  {"text": str(发给LLM), "sources": [str], "contexts": dict(可选), "handoff": bool}

安全边界 (阶段二):
  - 只读/查询工具开放给 Agent (retrieve_knowledge / query_products /
    check_suitability / query_holdings)。
  - 资金动账(申购/赎回/撤单)不注册为 Agent 工具 → 由 cs_graph 的确定性门控节点
    (subscribe_node) 执行并留痕, 机器不赌 LLM 自觉放行。
"""

import re
import logging

from src.business.finance_services import (
    query_products, check_suitability, query_holdings,
    query_account_balance, query_transactions, query_credit_card,
    query_loan, query_asset_overview, query_order_status,
)
from src.core.slot_filler import KNOWN_PRODUCTS

logger = logging.getLogger(__name__)


# 产品简称 (与 data/finance_kb/catalog.jsonl 对齐; 供"大额存单/结构性存款"这类口语命中)
_PRODUCT_ALIASES = {
    "大额存单3年期": "大额存单",
    "结构性存款·挂钩黄金3个月": "结构性存款",
    "稳盈添利30天": "稳盈添利",
    "安心固收90天": "安心固收",
    "平衡增利180天": "平衡增利",
    "私银聚享混合策略": "私银聚享",
    "日日盈现金管理类": "日日盈",
    "安鑫纯债基金": "安鑫纯债",
    "远见成长混合基金": "混合基金",
    "盛世稳赢增额终身寿": "增额终身寿",
    "安享颐年养老年金": "养老年金",
}


def _query_product_names(query: str) -> list:
    """返回查询中命中的目录产品名 (全名或简称命中)"""
    hits = []
    for name in KNOWN_PRODUCTS:
        # 别名表只覆盖部分产品, 缺省值必须是 None 而不是 "" ——
        # `"" in query` 恒为真, 会让别名表没覆盖的产品在**任何**提问上都"命中",
        # 于是「存款保险保不保理财产品」被判成点名了 13 个产品,
        # 下面 retrieve_knowledge 的对齐检查就把检索结果整个丢掉, sources 全是空的。
        alias = _PRODUCT_ALIASES.get(name)
        if name in query or (alias and alias in query):
            hits.append(name)
    return hits


def _fmt_local(local: list) -> tuple:
    """格式化检索结果, 返回 (文本行, 来源列表)"""
    lines = []
    sources = []
    for c in local:
        src = c.get("source", "未知")
        lines.append(f"- {c.get('content', '')[:1500]} [来源: {src}]")
        if src and src != "未知" and src not in sources:
            sources.append(src)
    return lines, sources


def format_kb_text(query: str, contexts: dict) -> dict:
    """把检索结果格式化成给模型看的原文, 并施加产品对齐守卫.

    守卫: 查询点名的产品若在检索内容里一个都没命中, 整份丢弃 (fail closed) ——
    防的是"拿别的产品的内容冒充它"。但判定必须**逐产品**且认口语简称:
      1) 命中判定用「目录全名 or 别名」—— 检索文本里写的通常是"大额存单"这种简称,
         只认目录全名会把明明找到的产品判成"未检索到";
      2) 只缺一个产品时保留已找到的内容(只标注缺的那个), 不做一票否决 ——
         否则问"A 和 B 哪个好"而只缺 B 时, 连 A 都答不了。

    独立成函数是为了让完整管道也能用上这道守卫 —— 图里 route_retrieve_node
    原本直接调 HybridRetriever, 绕过了它。
    """
    local = (contexts or {}).get("local") or []
    lines, sources = _fmt_local(local)
    if lines:
        text = "【知识库检索结果】\n" + "\n".join(lines)
    else:
        text = "【知识库检索结果】\n(未检索到相关文档)"

    names = _query_product_names(query)
    if names:
        all_content = " ".join((c.get("content") or "") for c in local)
        found, missing = [], []
        for name in names:
            alias = _PRODUCT_ALIASES.get(name)
            if name in all_content or (alias and alias in all_content):
                found.append(name)
            else:
                missing.append(name)
        if not found:
            return {"text": (f"【知识库检索结果】\n知识库中未检索到「{'、'.join(missing)}」的相关内容，"
                             f"命中的内容与其无关、不应采用。如仍无法获取，请如实告知用户"
                             f"知识库暂无该产品资料，不要用其他内容冒充，可建议转人工坐席。"),
                    "sources": []}
        if missing:
            text = (f"【注意】知识库中未检索到「{'、'.join(missing)}」的相关内容，"
                    f"不要用其他产品的内容替代它作答，可建议转人工坐席。\n" + text)
    return {"text": text, "sources": sources}


def retrieve_knowledge(query: str, top_k: int = 5) -> dict:
    """检索理财知识库 (产品说明书/政策条款/规则口径), 返回格式化原文 + 来源"""
    from src.retrieval.retriever import HybridRetriever
    res = HybridRetriever().hybrid_retrieve(query, top_k=top_k)
    out = format_kb_text(query, res)
    return {"text": out["text"], "sources": out["sources"], "contexts": res}


TOOL_DEFINITIONS = [
    {
        "name": "retrieve_knowledge",
        "description": "检索理财知识库（产品说明书/政策条款/收益口径/费率），返回相关原文片段（含来源）。回答产品事实、政策、口径问题前优先调用。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要检索的理财查询（可含产品名、政策、口径关键词）"}},
            "required": ["query"],
        },
    },
    {
        "name": "query_products",
        "description": "按产品类型/风险等级/关键词查询当前在售产品清单（结构化实时目录）。用户问“有哪些产品/推荐/R2 的/30天理财”时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "product_type": {"type": "string", "description": "产品类型，如 理财/存款/基金/保险/现金管理 (可选)"},
                "risk": {"type": "string", "description": "风险等级 R1~R5 (可选)"},
                "keyword": {"type": "string", "description": "产品名或类型关键词 (可选)"},
            },
        },
    },
    {
        "name": "check_suitability",
        "description": "校验某产品与当前用户风险测评等级是否匹配（适当性，确定性判定）。用户问“我能不能买/适合买 X/R3 我能买吗”时调用。",
        "parameters": {
            "type": "object",
            "properties": {"product_id_or_name": {"type": "string", "description": "产品名或产品ID，如 平衡增利180天 / P004"}},
            "required": ["product_id_or_name"],
        },
    },
    {
        "name": "query_holdings",
        "description": "查询当前登录用户的持仓（只读本人数据）。用户问“我的持仓/我买了什么/查我的理财”时调用。",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "query_account_balance",
        "description": "查询当前用户账户余额（储蓄卡+定期存款）。用户问“我卡里有多少钱/余额多少/定期存款”时调用。生产环境对接银行核心账户系统。",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "query_transactions",
        "description": "查询当前用户最近交易明细（收入/支出/理财交易）。用户问“最近交易/流水/收支明细/钱花哪了”时调用。生产环境对接交易流水系统。",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "返回条数，默认10"}}},
    },
    {
        "name": "query_credit_card",
        "description": "查询当前用户信用卡信息（额度/账单/还款日）。用户问“信用卡账单/额度多少/什么时候还款”时调用。生产环境对接信用卡核心系统。",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "query_loan",
        "description": "查询当前用户贷款信息（剩余本金/月供/利率）。用户问“贷款还剩多少/月供多少/房贷余额”时调用。生产环境对接信贷管理系统。",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "query_asset_overview",
        "description": "查询当前用户资产总览（存款+理财+基金-负债=净资产）。用户问“我总资产多少/资产配置/净资产”时调用。生产环境对接数据中台/统一客户视图。",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "query_order_status",
        "description": "查询当前用户理财订单状态（申购/赎回处理进度）。用户问“申购成功了吗/赎回什么时候到账/订单状态”时调用。生产环境对接理财业务系统。",
        "parameters": {"type": "object", "properties": {}},
    },
]


EXECUTE_TOOLS = {
    "retrieve_knowledge": retrieve_knowledge,
    "query_products": query_products,
    "check_suitability": check_suitability,
    "query_holdings": query_holdings,
    # ── 银行核心系统对接预留接口 (只读查询, Mock实现) ──
    "query_account_balance": query_account_balance,
    "query_transactions": query_transactions,
    "query_credit_card": query_credit_card,
    "query_loan": query_loan,
    "query_asset_overview": query_asset_overview,
    "query_order_status": query_order_status,
}

# P8: 所有工具调用经安全拦截器 (越权防护 + 审计)
from src.security.interceptor import intercept
EXECUTE_TOOLS = {name: intercept(name, fn) for name, fn in EXECUTE_TOOLS.items()}
