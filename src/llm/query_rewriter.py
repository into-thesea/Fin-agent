"""
查询补全 (理财客服) — 规则式产品指代补全, 无 LLM、无领域外内容。

理财多轮的上下文(产品/金额/测评等级)已由 dialog_state.slots 跨轮累积,
领域节点回答前另有记忆注入; 这里只补一件安全的事:
  下一句用 "它 / 这款 / 这个" 等近指词指向上一轮产品时, 替换为具体产品名,
  让路由与检索绑定到同一款产品。

凡无法安全确定的引用一律不猜、原样返回; 绝不编造数字或领域外内容,
也因此不需要 LLM 兜底 (旧版靠 LLM 自由生成, 与理财域错位)。

**唯一实现**: 线上 chat.py 与评测脚本 eval_retrieval.py --with-rewrite 都走这一份,
不要再另起一份 —— 评测必须量的是线上真正跑的改写, 否则指标没有意义。
"""

import logging

logger = logging.getLogger(__name__)

# 指向上一轮产品的近指/指代词 (按长度优先整体替换; 不含泛指"那/该/他/她", 避免误绑)
_PRODUCT_REFERENCE = [
    "这个产品", "这款产品", "这款", "这个", "它们", "它",
]

# 产品别名 → 规范名 反向映射 (从 knowledge_graph.retriever 同步, 最长优先匹配)
# 构建一次, 避免每次调用都遍历
def _build_alias_to_canonical() -> dict:
    """别名(含全名) → 规范产品名, 按别名长度降序供最长优先替换"""
    try:
        from src.knowledge_graph.retriever import _PRODUCT_ALIASES
    except Exception:
        return {}
    m = {}
    for canonical, aliases in _PRODUCT_ALIASES.items():
        m[canonical] = canonical  # 全名也加入, 避免重复替换
        for a in aliases:
            if a and a not in m:  # 短别名不覆盖长别名的映射
                m[a] = canonical
    return dict(sorted(m.items(), key=lambda kv: -len(kv[0])))

_ALIAS_TO_CANONICAL = _build_alias_to_canonical()


def _focal_product(state) -> str:
    """当前会话焦点产品 (取自持久化槽位; state 无 slots/缺省时返回空)"""
    slots = getattr(state, "slots", None) or {}
    return slots.get("product") or ""


def _expand_product_aliases(query: str) -> str:
    """把用户问题中的产品口语别名替换为规范产品名, 提升检索精确匹配.

    最长优先匹配, 避免短别名误替换 (如"国债"不替换"国债10年期"中已含规范名的部分).
    已含规范名时不重复替换. 纯规则, 零 LLM.
    """
    if not query or not _ALIAS_TO_CANONICAL:
        return query
    result = query
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if alias == canonical:
            continue  # 全名不需要替换
        if alias in result and canonical not in result:
            result = result.replace(alias, canonical)
            logger.info("别名扩展: '%s' → '%s' (规范名 %s)", alias, result, canonical)
    return result


def rewrite_query(query: str, state) -> str:
    """补全指向上一轮产品的省略查询 + 产品别名扩展为规范名.

    规则式、零 LLM、零编造。无历史 / 无焦点产品 / 已含产品名 / 不含指代近指词 时原样返回。
    别名扩展在指代补全之后执行, 确保补全后的产品名也能被识别.
    """
    q = (query or "").strip()
    if not q:
        return query

    # 1) 指代补全 (多轮上下文: "它" → 上一轮产品名)
    rewritten = q
    if getattr(state, "history", None):
        product = _focal_product(state)
        if product and product not in rewritten:
            for token in _PRODUCT_REFERENCE:
                if token in rewritten:
                    rewritten = rewritten.replace(token, product)
                    logger.info("查询补全: '%s' → '%s'", query, rewritten)

    # 2) 别名扩展 (口语叫法 → 规范产品名, 提升检索命中)
    expanded = _expand_product_aliases(rewritten)

    return expanded if expanded != query else query
