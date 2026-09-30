"""
查询补全 (理财客服) — 规则式产品指代补全, 无 LLM、无领域外内容。

理财多轮的上下文(产品/金额/测评等级)已由 dialog_state.slots 跨轮累积,
领域节点回答前另有记忆注入; 这里只补一件安全的事:
  下一句用 "它 / 这款 / 这个" 等近指词指向上一轮产品时, 替换为具体产品名,
  让路由与检索绑定到同一款产品。

凡无法安全确定的引用一律不猜、原样返回; 绝不编造数字或领域外内容,
也因此不需要 LLM 兜底 (旧版靠 LLM 自由生成, 与理财域错位)。
"""

import logging

logger = logging.getLogger(__name__)

# 指向上一轮产品的近指/指代词 (按长度优先整体替换; 不含泛指"那/该/他/她", 避免误绑)
_PRODUCT_REFERENCE = [
    "这个产品", "这款产品", "这款", "这个", "它们", "它",
]


def _focal_product(state) -> str:
    """当前会话焦点产品 (取自持久化槽位; state 无 slots/缺省时返回空)"""
    slots = getattr(state, "slots", None) or {}
    return slots.get("product") or ""


def rewrite_query(query: str, state) -> str:
    """补全指向上一轮产品的省略查询。

    规则式、零 LLM、零编造。无历史 / 无焦点产品 / 已含产品名 / 不含指代近指词 时原样返回。
    """
    q = (query or "").strip()
    if not q or not getattr(state, "history", None):
        return query
    product = _focal_product(state)
    if not product:
        return query
    if product in q:
        return query  # 已带焦点产品名, 不重复注入

    rewritten = q
    for token in _PRODUCT_REFERENCE:
        if token in rewritten:
            rewritten = rewritten.replace(token, product)
            logger.info("查询补全: '%s' → '%s'", query, rewritten)
    return rewritten if rewritten != query else query
