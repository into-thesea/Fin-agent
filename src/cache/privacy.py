"""
缓存隐私守卫 — 金融域用户私有 / 资金动账类问答不进入共享缓存

两类必须跳过 (读写两侧都拒绝, 防跨用户串答 / 隐私泄漏):
  1. 用户私有: 我的持仓/挂失/验证码/密码/登录/账号/卡号等
  2. 资金动账: 明确动作指令 + 金额 (我要买5万 / 申购X / 撤单) 或携带 6 位验证码

通用规则类 (理财怎么买/能随时赎回吗) 不含用户信号, 仍可缓存。
"""

import re

_CODE_RE = re.compile(r"\b\d{6}\b")
_AMOUNT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:万|w|W|元)")

_PRIVATE_WORDS = [
    "我的", "持仓", "挂失", "验证码", "密码", "登录", "账号", "卡号", "余额",
]

_MONEY_VERBS = ["买", "申购", "购买", "赎回", "撤单", "转出", "转入", "提现"]
_ACT_WORDS = ["我要", "我想", "帮我", "请", "立刻", "现在", "马上"]


def is_private_or_money_query(query: str) -> bool:
    q = query or ""
    if any(w in q for w in _PRIVATE_WORDS):
        return True
    if _CODE_RE.search(q):
        return True
    # 动账动作 + (金额 或 指令化表述) → 用户私有操作
    if any(v in q for v in _MONEY_VERBS) and (_AMOUNT_RE.search(q) or any(x in q for x in _ACT_WORDS)):
        return True
    return False
