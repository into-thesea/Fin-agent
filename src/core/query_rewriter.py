"""
Query Rewriting — 检索前的用户问题改写

解决多轮对话中的指代、省略、口语化问题，提升检索召回率。

两层策略:
  1. 规则层 (零延迟): 代词替换 + 省略补全，利用 slots 中的产品名
  2. LLM 层 (可选): 复杂情况用轻量模型结合对话历史改写

接入位置: cs_graph.py route_retrieve_node 中，检索前调用 rewrite_query()。
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 代词模式: 它/这个/那个/这款/该产品/该理财/该基金
_PRONOUN_PATTERN = re.compile(r"(它|这个|那个|这款|该(?:产品|理财|基金|保险|存款))")

# 省略模式: 单独的 "呢/吗/怎么样/如何/多少/什么" 结尾，且句中无产品名
_BARE_QUESTION = re.compile(r"^(呢|吗|怎么样|如何|多少|什么|为啥|为什么|靠谱吗|安全吗)$")


def _rule_rewrite(query: str, slots: dict) -> Optional[str]:
    """规则层改写: 代词替换 + 省略补全

    Returns:
        改写后的 query，或 None (规则未命中)
    """
    slots = slots or {}
    product = slots.get("product") or slots.get("product_name")
    if not product:
        return None

    rewritten = query

    # 1. 代词替换: "它的风险等级" → "稳盈添利30天的风险等级"
    if _PRONOUN_PATTERN.search(rewritten):
        rewritten = _PRONOUN_PATTERN.sub(product, rewritten, count=1)
        logger.info("Query改写(代词替换): %s → %s", query, rewritten)

    # 2. 省略补全: "收益呢" → "稳盈添利30天的收益是多少"
    #    仅当句中没有产品名且很短时触发
    if len(query) <= 15 and product not in query:
        # "呢/吗/怎么样" 结尾的短句
        if re.search(r"(呢|吗|怎么样|如何|多少|什么)$", query):
            rewritten = f"{product}的{query}"
            logger.info("Query改写(省略补全): %s → %s", query, rewritten)

    if rewritten != query:
        return rewritten
    return None


def _llm_rewrite(query: str, slots: dict, dialog_history: list = None) -> str:
    """LLM 层改写: 结合对话历史补全省略和指代

    仅在规则层未命中时调用，用 cheap 模型控制成本。
    """
    from src.llm.llm_client import create_client

    product = (slots or {}).get("product", "")
    history_text = ""
    if dialog_history:
        # 只保留最近 3 轮的用户问题
        user_qs = [m.get("content", "") for m in dialog_history[-6:]
                   if m.get("role") == "user"][-3:]
        if user_qs:
            history_text = "历史问题:\n" + "\n".join(f"- {q}" for q in user_qs)

    prompt = f"""你是金融客服的查询改写助手。将用户当前问题改写成独立、完整、适合知识库检索的标准问题。

规则:
- 补全省略的产品名（如果上下文有）
- 消解代词（它/这个/那个 → 具体产品名）
- 口语化表达转为标准表述
- 只输出改写后的问题，不要解释

{history_text}
当前焦点产品: {product or '未知'}
用户问题: {query}
改写后:"""

    try:
        client = create_client(cheap=True)
        result = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        rewritten = result.strip().strip('"').strip("'")
        if rewritten and rewritten != query:
            logger.info("Query改写(LLM): %s → %s", query, rewritten)
            return rewritten
    except Exception as e:
        logger.warning("Query改写 LLM 调用失败，使用原问题: %s", str(e)[:80])

    return query


def rewrite_query(query: str, slots: dict = None,
                  dialog_history: list = None, use_llm: bool = True) -> str:
    """检索前的问题改写入口

    Args:
        query: 用户原始问题
        slots: 会话槽位 (含 product 等)
        dialog_history: 对话历史 (可选)
        use_llm: 是否启用 LLM 层改写 (默认 True)

    Returns:
        改写后的问题。规则层优先，未命中时走 LLM 层，都失败返回原问题。
    """
    if not query or not query.strip():
        return query

    # 规则层 (零延迟)
    rule_result = _rule_rewrite(query, slots or {})
    if rule_result:
        return rule_result

    # LLM 层 (可选，有额外延迟)
    if use_llm:
        return _llm_rewrite(query, slots or {}, dialog_history)

    return query
