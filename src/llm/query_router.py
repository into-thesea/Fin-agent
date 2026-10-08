"""
Fin-Agent → 智能客服 规则路由 (v4.0)

只做一件事: 区分 greeting 与 complex。
  - greeting: 问候/闲聊 → 轻量 LLM, 无检索, 极速
  - complex:  其余全部 → 进图, 由五道闸决定快答还是完整管道

**难度判定不再由本模块的关键词表承担。** v3.0 曾用 _FAQ_MARKERS(51 词) /
_COMPLEX_MARKERS / _PRODUCT_NAMES(15 词) 三张手写词表猜"问题简不简单", 结果:
  - 词表从 30 词涨到 51 词, 吞掉了 `风险/安全/靠谱/产品/理财` 这些域词,
    导致 golden 里标为 fraud_report 的 6 条诈骗问题全部走快速通道,
    而 HANDOFF_INTENTS 的确定性转人工写在 orchestrator 里、快通道根本到不了;
  - "区别" 从 _COMPLEX_MARKERS 挪进 _FAQ_MARKERS, 把 product_compare 全推进了快通道。
判据改为「分类器有多确定 + KB 里有没有依据」, 见 src/core/fast_path.py
与 docs/superpowers/specs/2026-10-03-fast-path-confidence-gate-design.md
"""

from typing import Literal

QueryClass = Literal["greeting", "complex"]

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


def classify(query: str) -> QueryClass:
    """只区分 greeting 与 complex (纯规则, 零成本, <0.1ms)。

    greeting 要求整句很短 —— 问候词出现在长句里 (如"你好，我想申购5万理财")
    说明用户有实际诉求, 不是寒暄。
    """
    q = (query or "").strip()
    if len(q) <= 12 and any(g in q.lower() for g in _GREETINGS):
        return "greeting"
    return "complex"
