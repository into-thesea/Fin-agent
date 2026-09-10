"""
合规检查 — 敏感词 / PII 掩码 / 不当承诺 + 金融风险提示

客服回答发布前检查:
  1. 广告法违禁词 (绝对化用语) + 金融禁语 (稳赚/零风险/保本高收益等承诺)
  2. PII 掩码: 手机号 / 身份证 / 银行卡号 一律不原样出站
  3. 非存款产品涉及收益/基准/年化的回答自动附标准风险提示 (finalize)

用法:
    from src.core.compliance import check, finalize
    c = check(answer)            # {clean, hits, pii, masked}
    answer = finalize(answer)    # 掩码敏感词/PII + 按需附风险提示
"""

import re

# 广告法绝对化用语 + 客服不当承诺
SENSITIVE_WORDS = [
    "绝对", "保证", "一定", "百分百", "100%", "100％",
    "终身", "最便宜", "最低价", "全网最低", "第一", "最好", "顶级",
    "不会出错", "包您满意", "无条件退款", "随时退款", "绝对安全",
]

# 金融禁语: 收益承诺 / 虚假安全性表述
FINANCE_BANNED_WORDS = [
    "稳赚不赔", "稳赚", "零风险", "无风险", "保本高收益", "保底收益",
    "包赚", "一定赚", "保收益", "保证收益", "保证年化", "躺着赚钱",
    "百分百获利", "承诺保本",
]

# 出站风险提示 (非存款产品涉收益必附)
DISCLAIMER = ("温馨提示：理财/基金/保险非存款，产品有风险，投资须谨慎；"
              "业绩比较基准、历史收益等仅为参考，不构成收益承诺。")

# PII 正则
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_CARD_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")


def _mask_pii(text: str) -> tuple:
    """掩码 PII, 返回 (masked_text, [命中标签])"""
    labels = []

    def _redact(pat, keep_head=3, keep_tail=4):
        nonlocal labels
        def _m(m):
            s = m.group(0)
            labels.append("phone" if pat is _PHONE_RE else ("id" if pat is _ID_RE else "card"))
            if len(s) <= keep_head + keep_tail:
                return "*" * len(s)
            return s[:keep_head] + "*" * (len(s) - keep_head - keep_tail) + s[-keep_tail:]
        return _m

    t = _PHONE_RE.sub(_redact(_PHONE_RE), text)
    t = _ID_RE.sub(_redact(_ID_RE), t)
    t = _CARD_RE.sub(_redact(_CARD_RE), t)
    return t, labels


def _mask_words(text: str, words: list) -> tuple:
    """掩码命中词, 返回 (masked_text, hits)"""
    hits = [w for w in words if w and w in text]
    masked = text
    for w in hits:
        masked = masked.replace(w, "*" * len(w))
    return masked, hits


def check(text: str) -> dict:
    """检查文本: 敏感词命中并掩码 (含 PII). 返回 {clean, hits, pii, masked}"""
    text = text or ""
    masked, hits = _mask_words(text, SENSITIVE_WORDS + FINANCE_BANNED_WORDS)
    masked, pii = _mask_pii(masked)
    return {
        "clean": not hits and not pii,
        "hits": hits,
        "pii": pii,
        "masked": masked,
    }


def risk_note_needed(text: str) -> bool:
    """判断是否需附金融风险提示: 提到非存款产品且涉收益口径, 但原文缺提示"""
    t = text or ""
    if not any(k in t for k in ("理财", "基金", "保险", "业绩比较基准")):
        return False
    if not any(k in t for k in ("收益", "基准", "年化", "利率", "增值", "返还")):
        return False
    if any(k in t for k in ("风险", "非存款", "不构成", "谨慎", "温馨提示", "提示")):
        return False
    return True


def finalize(text: str) -> str:
    """出站处理: 掩码敏感词/PII + 按需附风险提示, 返回最终对外文本"""
    out = check(text)["masked"]
    if risk_note_needed(text):
        out = out.rstrip() + "\n\n" + DISCLAIMER
    return out
