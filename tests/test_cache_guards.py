"""理财域 L2/L3 守卫 — 防跨意图串答, 同时别把真同义也挡掉

背景: L2 的原有守卫 (公司名/年份) 是为年报问答建的, 在理财产品域全部空转
(_extract_company 对"稳盈添利30天"返回 None)。裸相似度下
"理财产品怎么赎回" vs "理财产品怎么申购" = 0.518 > 阈值 0.50, 会拿申购的答案回答赎回。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.core.answer_cache import _action_groups, _extract_product, _l2_similarity


# ── 动作守卫 ────────────────────────────────────

def test_opposite_actions_blocked():
    """申购 ≠ 赎回: 这是最容易串答的一对"""
    assert _l2_similarity("理财产品怎么赎回", "理财产品怎么申购") == 0.0


def test_same_action_not_blocked():
    """买 / 购买 / 申购 属同一组, 不能被动作守卫挡掉"""
    assert _action_groups("我想买理财") & _action_groups("我想购买理财产品")
    assert _l2_similarity("安鑫纯债基金怎么申购", "安鑫纯债基金如何购买") > 0.0


def test_multi_action_query_not_blocked():
    """一句里同时有申购和撤单 → 与另一句有交集, 不应被挡 (保守)"""
    assert _action_groups("申购后能撤单吗") & _action_groups("怎么申购")


# ── 产品名守卫 ──────────────────────────────────

def test_product_names_loaded_from_catalog():
    names = [_extract_product(q) for q in
             ["天天利货币基金收益", "安鑫纯债基金风险", "稳盈添利30天期限"]]
    assert all(names), f"产品名应能从 catalog.jsonl 识别出来, 实得 {names}"
    assert len(set(names)) == 3


def test_different_products_blocked():
    assert _l2_similarity("天天利货币基金收益", "安鑫纯债基金收益") == 0.0
    assert _l2_similarity("稳盈添利30天期限", "平衡增利180天期限") == 0.0


def test_generic_query_without_product_not_blocked():
    """没点具体产品的泛问不该被产品守卫挡掉"""
    assert _extract_product("理财产品怎么赎回") is None
    assert _l2_similarity("大额存单安全吗", "存款保险保什么") >= 0.0


# ── 阈值与守卫的一致性 ──────────────────────────

def test_l3_threshold_in_safe_gap():
    """L3 阈值必须落在"误命中上界"与"真同义下界"之间。
    实测误命中最高 0.771, 真同义最低 0.911 —— 阈值须 >0.78 且 <0.91。"""
    from src.core.answer_cache import L3_THRESHOLD
    assert 0.78 < L3_THRESHOLD < 0.911
