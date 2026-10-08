"""L0 产品域由 catalog 派生 —— 不许在手写产品词表里腐烂.

旧实现 (_intent_to_l0) 拿关键词扫 product_type, 认不出的全落兜底 "wealth",
连"固定收益"(词表里写的是"固收")和"现金管理类属于理财产品"都是靠撞对/撞错。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.slot_filler import (
    _UNMAPPED_TYPES,
    CATALOG,
    L0_DOMAINS,
    intent_to_l0,
)

# 7 个产品域 (common/out_of_scope 是兜底域, 不算"推出了具体域")
PRODUCT_DOMAINS = set(L0_DOMAINS) - {"common", "out_of_scope"}


def test_catalog_type_coverage():
    """catalog 新增产品类别时必须补 _L0_BY_TYPE, 不能静默落到 common"""
    assert not _UNMAPPED_TYPES, f"catalog 有未映射 L0 的类别: {sorted(_UNMAPPED_TYPES)}"


def test_every_product_resolves_to_a_real_domain():
    """每个真实产品都必须推出具体域; 落到 common 说明映射漏了"""
    for p in CATALOG:
        l0 = intent_to_l0("product_consult", {"product": p["name"]})
        assert l0 in PRODUCT_DOMAINS, f'{p["name"]} (type={p["type"]}) 推出了 {l0}'


def test_known_products():
    cases = {
        "大额存单3年期": "deposit",
        "结构性存款·挂钩黄金3个月": "deposit",
        "日日盈现金管理类": "wealth",   # KB: 现金管理类产品属于理财产品, 非存款
        "稳盈添利30天": "wealth",       # type=固定收益, 旧实现靠兜底撞对
        "安鑫纯债基金": "fund",
        "天天利货币基金": "fund",       # 旧实现见"货币"会误判存款
        "盛世稳赢增额终身寿": "insurance",
        "储蓄国债(电子式)3年期": "bond",
        "固定收益类集合信托计划": "trust",
        "账户黄金(积存金)": "gold",
    }
    for name, want in cases.items():
        got = intent_to_l0("product_consult", {"product": name})
        assert got == want, f"{name}: {got} != {want}"


def test_unknown_is_common_not_wealth():
    """判不出就说判不出: 旧的默认 wealth 会给基金/存款问题套上理财合规话术"""
    assert intent_to_l0("product_consult", {}) == "common"
    assert intent_to_l0("product_consult", {"product_type": "外汇(某新类别)"}) == "common"


def test_intent_drives_common_domains():
    for intent in ("deposit_insurance", "fraud_report", "complaint", "service_policy", "unknown"):
        assert intent_to_l0(intent, {}) == "common"


if __name__ == "__main__":
    for fn in (test_catalog_type_coverage, test_every_product_resolves_to_a_real_domain,
               test_known_products, test_unknown_is_common_not_wealth,
               test_intent_drives_common_domains):
        fn()
    print("OK: L0 derives from catalog")
