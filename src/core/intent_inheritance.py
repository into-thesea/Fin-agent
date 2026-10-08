"""续轮意图继承 —— 确定性规则, 零 LLM 成本

在路由 LLM 之前判断当前问题是否为上一轮流程的延续, 若是则直接沿用上一轮意图,
跳过路由 LLM 调用 (命中时延迟为负贡献)。

优先级 (硬约束):
    强信号规则 (诈骗/投诉直出) > 话题切换检测 > 本继承规则 > LLM 路由

设计原则:
    只对"明确的流程延续信号"做继承, 不做宽泛猜测;
    咨询类意图 (product_consult / income_question / deposit_insurance 等) 的续轮
    仍交给 LLM 判断, 避免错误传染。
"""
import re
from typing import Optional, Tuple

# ── 话题切换关键词: 命中则不继承 (明确开启新业务域或新问询类型) ──
_TOPIC_SWITCH_KEYWORDS = (
    "存款保险", "诈骗", "被骗", "投诉", "客服", "人工", "转人工",
    "基金", "保险", "黄金", "国债", "信托", "贷款", "信用卡",
    "手续费", "费率", "收益", "利率", "利息",
    "本金", "保本", "靠谱", "可靠", "安全吗", "风险",
)

# ── 确认类词 (流程延续) ──
_CONFIRM_WORDS = ("确认", "好的", "可以", "对", "嗯", "行", "没问题", "好", "是")

# ── 验证码/校验码模式 ──
_CODE_PATTERN = re.compile(r"(验证码|校验码|动态码|短信码)\s*\d{4,8}")

# ── 纯金额/纯数字模式 (允许 "50万" "5000" "5万" "3.5万") ──
_AMOUNT_PATTERN = re.compile(r"^\s*\d+(\.\d+)?\s*(万|元|块|w|W)?\s*$")

# ── 开启新话题的疑问词 (短问法继承时排除) ──
_NEW_TOPIC_QUESTION = re.compile(r"(什么是|怎么|为什么|如何|哪些|哪里|有没有|是否)")

# ── 允许短问法继承的流程类意图 ──
_FLOW_INTENTS = {"buy_process", "hold_redeem"}


def is_topic_switch(query: str) -> bool:
    """检测当前问题是否开启了新话题 (命中则不继承)"""
    q = query.strip()
    for kw in _TOPIC_SWITCH_KEYWORDS:
        if kw in q:
            return True
    return False


def try_inherit_intent(
    query: str,
    last_intent: Optional[str],
    awaiting_slot: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    尝试沿用上一轮意图。

    Args:
        query: 当前用户问题 (指代改写后的)
        last_intent: 上一轮意图 (12 类平面结构), 首轮为 None
        awaiting_slot: 上一轮待补充的槽位名 (如 "amount")

    Returns:
        (是否继承, 继承的意图); 不继承时返回 (False, None)
    """
    if not last_intent:
        return False, None

    q = query.strip()
    if not q:
        return False, None

    # 话题切换 → 不继承
    if is_topic_switch(q):
        return False, None

    # 规则 a: 验证码/校验码 + 上一轮是申购 → 延续 buy_process
    if _CODE_PATTERN.search(q) and last_intent == "buy_process":
        return True, "buy_process"

    # 规则 b: 确认词 + 上一轮是流程类意图 → 沿用
    if last_intent in _FLOW_INTENTS:
        for w in _CONFIRM_WORDS:
            if q == w or q.startswith(w):
                return True, last_intent

    # 规则 c: 纯金额/纯数字 + 上一轮在等 amount 槽位 → 沿用
    if awaiting_slot == "amount" and _AMOUNT_PATTERN.match(q):
        return True, last_intent

    # 规则 d: 短问法 (去标点 < 10 字) + 无新话题疑问词 + 上一轮是流程类意图 → 沿用
    # 仅对流程类意图生效, 咨询类续轮仍交 LLM, 避免错误传染;
    # 阈值保守 (<10字), 防止"那我的本金还能拿回来吗"(11字)这类实质切换被误继承
    if last_intent in _FLOW_INTENTS:
        q_clean = re.sub(r"[，。？！、\s]", "", q)
        if len(q_clean) < 10 and not _NEW_TOPIC_QUESTION.search(q):
            return True, last_intent

    return False, None
