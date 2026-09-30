"""
Fin-Agent → 智能客服 规则路由 (v3.0 客服版)

把查询分为 3 类:
  - greeting:   问候/闲聊 → 轻量 LLM, 无检索, 极速 (<1s)
  - simple_fact: 简单 FAQ → 检索 + 轻量 LLM (~5s)
  - complex:    业务/复杂 → 完整 Pipeline (意图路由 + 工具循环)

通过纯规则 (正则 + 关键词) 分类, 零延迟, 无 LLM 调用。
"""

import re
from typing import Literal

QueryClass = Literal["greeting", "simple_fact", "complex"]

# ── 问候/闲聊模式 ─────────────────────────────
_GREETINGS = [
    "你好", "您好", "hello", "hi", "hey",
    "谢谢", "感谢", "多谢", "谢谢了", "辛苦了",
    "再见", "拜拜", "bye", "goodbye",
    "早上好", "下午好", "晚上好", "晚安",
    "在吗", "在不在", "有人吗",
    "好的", "明白了", "知道了", "了解",
    "嗯", "哦", "ok", "okay", "好的吧",
]

# ── 静态 FAQ 特征词 (简单事实/政策 → 快速通道) ──
_FAQ_MARKERS = [
    "怎么", "如何", "多少", "多少钱", "几天", "多久", "能不能", "可以吗",
    "是不是", "可不可以", "是什么", "什么是", "有哪些", "包括",
    "保本", "收益", "年化", "利息", "利率", "费率", "手续费", "费用",
    "期限", "到期", "净值", "风险等级", "测评", "起购", "门槛", "冷静期",
    "双录", "存款保险", "政策", "条款", "标准", "规则", "条件", "要求",
    "风险", "安全", "可靠", "靠谱", "区别", "不同", "差异",
    "产品", "理财", "基金", "存款", "保险", "大额存单", "结构性存款",
]

# ── 业务/动态/用户相关请求 → 完整管道 (我的/动账/转人工/对比分析) ──
_COMPLEX_MARKERS = [
    "我的", "我想", "我要", "帮我", "查一下", "查查", "查询", "持仓",
    "申购", "购买", "赎回", "挂失", "申请", "提交", "验证码",
    "投诉", "举报", "人工", "转人工", "不满意", "太差",
    "为什么", "怎么办", "怎么处理", "对比", "分析", "推荐", "建议",
    "适合我", "选哪个", "买哪个", "哪个好",
]

# 产品名关键词 (用于"单产品+单一属性"快速通道检测)
_PRODUCT_NAMES = [
    "稳盈添利", "日日盈", "现金管理", "安心固收", "平衡增利", "私银",
    "大额存单", "结构性存款", "纯债", "混合基金", "增额", "终身寿",
    "年金", "安鑫",
]


def classify(query: str) -> QueryClass:
    """
    快速分类查询 (纯规则, 零成本, <0.1ms)

    Returns:
        "greeting" | "simple_fact" | "complex"
    """
    q = query.strip()

    has_complex = any(re.search(m, q) for m in _COMPLEX_MARKERS)

    # 1. 问候/闲聊 (不能含业务请求词, 如 "你好我的订单" 不是纯问候)
    if not has_complex and any(g in q.lower() for g in _GREETINGS):
        return "greeting"

    # 2. 简单 FAQ (含 FAQ 特征词 + 无业务词 + 查询不太长)
    has_faq = any(m in q for m in _FAQ_MARKERS)
    # 单产品+单一属性检测: 查询提到一款具体产品且问一个属性, 可走快速通道
    has_single_product = any(p in q for p in _PRODUCT_NAMES)
    # 长度限制从40放宽到80: 很多真实用户问题在40-80字之间, 但仍是简单FAQ
    if has_faq and not has_complex and len(q) < 80:
        return "simple_fact"
    # 单产品咨询 (如 "稳盈添利30天收益怎么样") 即使不含FAQ标记词也走快速通道
    if has_single_product and not has_complex and len(q) < 80:
        return "simple_fact"

    # 3. 兜底: complex
    return "complex"
