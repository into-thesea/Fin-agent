"""
情绪处理 — 轻量级关键词规则 + 对话轮次统计

不引入情绪识别模型, 用关键词规则 + System Prompt 注入实现.
四类情绪: normal / anxiety / complaint / confusion

设计原则:
  - 零额外依赖, 纯规则匹配
  - 检测结果只用于调整 System Prompt, 不改变业务逻辑
  - 连续负面轮次自动升级共情和转人工建议
"""

import logging

logger = logging.getLogger(__name__)

# 焦虑关键词: 用户担心亏损/本金安全
ANXIETY_KEYWORDS = [
    "亏了", "亏损", "亏本金", "会不会亏", "会不会赔", "本金安全",
    "担心", "怕", "害怕", "睡不着", "焦虑", "紧张", "不踏实",
    "保本吗", "安全吗", "靠谱吗", "会不会跑路",
]

# 投诉/不满关键词
COMPLAINT_KEYWORDS = [
    "投诉", "骗人", "骗子", "太差", "垃圾", "坑爹", "忽悠",
    "不满", "不满意", "什么玩意", "什么东西", "态度差",
    "曝光", "举报", "银保监会", "12378",
]

# 困惑关键词
CONFUSION_KEYWORDS = [
    "没听懂", "听不懂", "什么意思", "再说一遍", "不明白",
    "看不懂", "不清楚", "怎么回事", "啥意思", "能解释一下",
]

# 负面情绪词 (用于累计 negative_turns)
NEGATIVE_KEYWORDS = ANXIETY_KEYWORDS + COMPLAINT_KEYWORDS


def detect_emotion(query: str, negative_turns: int = 0) -> str:
    """
    检测用户情绪

    Args:
        query: 用户问题
        negative_turns: 连续负面轮次数 (来自 DialogState)

    Returns:
        "normal" | "anxiety" | "complaint" | "confusion"
    """
    q = query or ""

    # 投诉优先级最高 (需要立即处理)
    for kw in COMPLAINT_KEYWORDS:
        if kw in q:
            return "complaint"

    # 焦虑
    for kw in ANXIETY_KEYWORDS:
        if kw in q:
            return "anxiety"

    # 困惑
    for kw in CONFUSION_KEYWORDS:
        if kw in q:
            return "confusion"

    # 连续 2 轮以上负面情绪, 即使本轮没有明确关键词也标记为焦虑
    if negative_turns >= 2:
        return "anxiety"

    return "normal"


def is_negative(query: str) -> bool:
    """判断本轮是否为负面情绪 (用于累计 negative_turns)"""
    q = query or ""
    return any(kw in q for kw in NEGATIVE_KEYWORDS)


def build_emotion_instruction(emotion: str) -> str:
    """
    根据情绪生成 System Prompt 追加指令

    Returns:
        追加到 System Prompt 的文本 (normal 时返回空字符串)
    """
    instructions = {
        "anxiety": "用户表现出焦虑或担忧，请先共情安抚再回答问题，回答中必须包含风险提示，语气要稳健 reassuring。",
        "complaint": "用户表达不满或投诉意向，请先致歉并表达理解，再解释问题，主动提供转人工坐席的选项。不要争辩或推卸责任。",
        "confusion": "用户表示困惑，请降低回答复杂度，用分步骤的方式解释，避免专业术语堆砌。",
        "normal": "",
    }
    return instructions.get(emotion, "")


def should_suggest_handoff(emotion: str, turn_count: int, negative_turns: int) -> bool:
    """
    是否应该建议转人工

    触发条件:
      - complaint 情绪
      - 同一话题超过 3 轮未解决
      - 连续 2 轮以上负面情绪
    """
    if emotion == "complaint":
        return True
    if negative_turns >= 2:
        return True
    if turn_count >= 3 and emotion in ("anxiety", "confusion"):
        return True
    return False
