import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pytest

from src.core.kb_settings import DEFAULTS
from src.core.kb_validate import ValidationError, validate


def _chunk(**over):
    return {**DEFAULTS["chunking"], **over}


def _retr(**over):
    return {**DEFAULTS["retrieval"], **over}


def test_defaults_pass():
    assert validate("chunking", _chunk())["chunk_size"] == 350
    assert validate("retrieval", _retr())["rrf_k"] == 60


@pytest.mark.parametrize("values,field", [
    (_chunk(chunk_size=100), "chunk_size"),          # 低于下限
    (_chunk(chunk_size=900), "chunk_size"),          # 高于上限
    (_chunk(overlap=350), "overlap"),                # == chunk_size, 必须严格小于
    (_chunk(overlap=400), "overlap"),                # > chunk_size
    (_chunk(max_chunk_content=400), "max_chunk_content"),  # > chunk_size(350)
    (_retr(top_k=0), "top_k"),
    (_retr(top_k=50), "top_k"),
    (_retr(mmr_lambda=1.5), "mmr_lambda"),
    (_retr(rrf_k=0), "rrf_k"),
    (_retr(rerank_model="gpt-4"), "rerank_model"),   # 不在白名单
    (_retr(top_k=float("inf")), "top_k"),            # JSON 的 1e400 会被解析成 inf
    (_chunk(max_chunk_content=float("inf")), "max_chunk_content"),
])
def test_rejects_out_of_range(values, field):
    section = "chunking" if field in DEFAULTS["chunking"] else "retrieval"
    with pytest.raises(ValidationError) as ei:
        validate(section, values)
    assert ei.value.field == field
    assert ei.value.rule          # 必须给出可读的原因, 不能是空串


def test_rerank_pool_must_exceed_top_k():
    """候选池不比最终条数多时 ApiReranker 会走"候选不足"的提前返回, 重排静默失效"""
    with pytest.raises(ValidationError) as ei:
        validate("retrieval", _retr(top_k=20, rerank_pool=10))
    assert ei.value.field == "rerank_pool"


def test_string_numbers_are_coerced():
    """前端表单常传字符串, 合法值应被接受并归一化成数字"""
    out = validate("retrieval", _retr(top_k="8", mmr_lambda="0.3"))
    assert out["top_k"] == 8 and isinstance(out["top_k"], int)
    assert out["mmr_lambda"] == 0.3 and isinstance(out["mmr_lambda"], float)


def test_unknown_section_rejected():
    with pytest.raises(ValidationError):
        validate("nope", {})


# ── 信任边界补充: 这些都是前端可能真的发过来的形状, 必须 400 而不是 500 ──

@pytest.mark.parametrize("bad", [
    ["gte-rerank-v2"],      # 列表: 不可哈希, 若直接 `v not in set` 会抛 TypeError
    None,
    {"name": "gte-rerank-v2"},
])
def test_rerank_model_non_string_rejected(bad):
    with pytest.raises(ValidationError) as ei:
        validate("retrieval", _retr(rerank_model=bad))
    assert ei.value.field == "rerank_model" and ei.value.rule


@pytest.mark.parametrize("section,field", [
    ("chunking", "chunk_size"),
    ("chunking", "overlap"),
    ("chunking", "max_chunk_content"),
    ("retrieval", "top_k"),
    ("retrieval", "rrf_k"),
    ("retrieval", "rerank_pool"),
])
def test_bool_is_not_a_number(section, field):
    """True 会被 int() 悄悄变成 1 —— 拒掉, 别让它伪装成合法值。

    断言 reason 里带"布尔": 否则 chunk_size=True 会因为 1 < 150 被范围检查顺带拒掉,
    用例就是绿的了却完全没碰到这个守卫。
    """
    base = _chunk if section == "chunking" else _retr
    with pytest.raises(ValidationError) as ei:
        validate(section, base(**{field: True}))
    assert ei.value.field == field
    assert "布尔" in ei.value.rule


def test_values_must_be_object():
    with pytest.raises(ValidationError):
        validate("retrieval", ["top_k"])


def test_unknown_field_rejected():
    with pytest.raises(ValidationError) as ei:
        validate("retrieval", _retr(chunk_size=400))
    assert ei.value.field == "chunk_size"


def test_as_dict_is_json_ready():
    """ValidationError 会直接作为 HTTP 400 的 detail 返回给前端"""
    with pytest.raises(ValidationError) as ei:
        validate("retrieval", _retr(top_k=0))
    d = ei.value.as_dict()
    assert d["field"] == "top_k" and d["value"] == 0 and d["rule"]


def test_result_is_detached_from_defaults():
    """返回值被调用方改写不该污染 DEFAULTS"""
    out = validate("chunking", _chunk())
    out["chunk_size"] = 999
    assert DEFAULTS["chunking"]["chunk_size"] == 350
