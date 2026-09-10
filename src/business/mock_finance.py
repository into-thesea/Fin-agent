"""
金融理财产品演示数据 (mock)

客户资料: 风险测评等级 + 演示短信验证码 + 持仓
产品主数据在 data/finance_kb/catalog.jsonl (唯一事实源, 由 core.slot_filler 加载后共享)。
"""

# 客户资料: investor_level 为风险测评等级 (决定适当性门控), verify_code 为演示短信验证码
PROFILES = {
    "u001": {"name": "演示客户甲", "investor_level": "R2", "verify_code": "123456"},
    "u002": {"name": "演示客户乙", "investor_level": "R3", "verify_code": "654321"},
    "u003": {"name": "演示客户丙", "investor_level": "R4", "verify_code": "111222"},
}

# 持仓: 金额单位 = 万元
HOLDINGS = {
    "u001": [
        {"product_id": "P002", "product_name": "稳盈添利30天", "amount_wan": 5.0,
         "term": "30天", "status": "持有中", "expected_end": "2026-09-28"},
        {"product_id": "P001", "product_name": "日日盈现金管理类", "amount_wan": 3.0,
         "term": "T+0", "status": "持有中", "expected_end": "-"},
    ],
    "u002": [
        {"product_id": "P004", "product_name": "平衡增利180天", "amount_wan": 10.0,
         "term": "180天", "status": "持有中", "expected_end": "2027-02-15"},
    ],
    "u003": [],
}


def get_profile(user_id: str):
    return PROFILES.get(user_id)


def add_holding(user_id: str, product: dict, amount_wan: float) -> None:
    """申购确认后写入持仓 (演示; 同一产品未到期不重复申购, 已存在则提示)"""
    exists = any(
        h["product_id"] == product["id"] and h["status"] == "持有中"
        for h in HOLDINGS.get(user_id, [])
    )
    if exists:
        return
    HOLDINGS.setdefault(user_id, []).append({
        "product_id": product["id"],
        "product_name": product["name"],
        "amount_wan": round(amount_wan, 2),
        "term": product.get("term", "-"),
        "status": "持有中",
        "expected_end": "-",
    })


def has_holding(user_id: str, product_id: str) -> bool:
    return any(h["product_id"] == product_id and h["status"] == "持有中"
               for h in HOLDINGS.get(user_id, []))
