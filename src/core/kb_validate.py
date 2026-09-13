"""
知识库参数校验 —— 纯函数, 无 I/O。

前端也会校验, 但那只为体验; 这里是唯一可信的一道, 不信任前端任何输入。
跨字段约束(overlap < chunk_size 等)也在这里, 因为它们只在同一 section 内可比。
"""
from __future__ import annotations

from src.core.kb_settings import DEFAULTS


class ValidationError(Exception):
    def __init__(self, field: str, value, rule: str):
        self.field, self.value, self.rule = field, value, rule
        super().__init__(f"{field}={value!r} 不合法: {rule}")

    def as_dict(self) -> dict:
        return {"field": self.field, "value": self.value, "rule": self.rule}


# 数值范围: (下限, 上限, 是否整数)
_RANGES = {
    "chunking": {
        "chunk_size": (150, 800, True),
        "overlap": (0, 400, True),
        "max_chunk_content": (100, 1000, True),
    },
    "retrieval": {
        "top_k": (1, 20, True),
        "mmr_lambda": (0.0, 1.0, False),
        "rrf_k": (1, 100, True),
        "rerank_pool": (5, 100, True),
    },
}
_BOOLS = {"chunking": set(), "retrieval": {"mmr_enabled", "rerank_enabled"}}
_STRINGS = {"chunking": set(), "retrieval": {"rerank_model"}}
_RERANK_MODELS = {"gte-rerank-v2"}


def _num(field, value, lo, hi, integral):
    # bool 会被 int() 悄悄变成 0/1 —— 那是个伪装的合法值, 先拦掉
    if isinstance(value, bool):
        raise ValidationError(field, value, f"必须是{'整数' if integral else '数字'}, 不能是布尔值")
    try:
        v = int(value) if integral else float(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(float('inf')) —— JSON 的 1e400 / Infinity 会被 json.loads 解析成 inf,
        # 不接住它就会以未捕获异常逃出本模块, 而拦截非法输入正是本模块存在的理由
        raise ValidationError(field, value, f"必须是{'整数' if integral else '数字'}")
    if not (lo <= v <= hi):
        raise ValidationError(field, value, f"必须在 {lo}~{hi} 之间")
    return v


def validate(section: str, values: dict) -> dict:
    """校验并归一化。通过返回完整 section, 失败抛 ValidationError。"""
    if section not in DEFAULTS:
        raise ValidationError("section", section, "未知配置段")

    if not isinstance(values, dict):
        raise ValidationError("values", values, "必须是对象")

    known = set(DEFAULTS[section]) | _BOOLS[section] | _STRINGS[section]
    for k in values:
        if k not in known:
            raise ValidationError(k, values[k], "不是该配置段的字段")

    out = dict(DEFAULTS[section])
    for k, v in values.items():
        if k in _BOOLS[section]:
            if not isinstance(v, bool):
                raise ValidationError(k, v, "必须是布尔值")
            out[k] = v
        elif k in _STRINGS[section]:
            # 先判类型: v 是 list/dict 时 `v not in set` 会抛 TypeError(500), 而这里要的是 400
            if not isinstance(v, str) or (k == "rerank_model" and v not in _RERANK_MODELS):
                raise ValidationError(k, v, f"只支持 {sorted(_RERANK_MODELS)}")
            out[k] = v
        else:
            lo, hi, integral = _RANGES[section][k]
            out[k] = _num(k, v, lo, hi, integral)

    # ── 跨字段约束 ──
    if section == "chunking":
        if out["overlap"] >= out["chunk_size"]:
            raise ValidationError("overlap", out["overlap"],
                                  f"必须小于 chunk_size({out['chunk_size']})")
        if out["max_chunk_content"] > out["chunk_size"]:
            raise ValidationError("max_chunk_content", out["max_chunk_content"],
                                  f"不能大于 chunk_size({out['chunk_size']})")
    if section == "retrieval":
        # 候选池不比最终条数多时, ApiReranker.rerank 会因 len(candidates) <= top_k
        # 走提前返回 —— 重排静默失效, 正是本项目一直在防的那类问题。
        if out["rerank_pool"] <= out["top_k"]:
            raise ValidationError("rerank_pool", out["rerank_pool"],
                                  f"必须大于 top_k({out['top_k']}), 否则重排不生效")

    return out
