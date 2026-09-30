"""
金融理财产品演示数据 (mock)

客户资料: 风险测评等级 + 演示短信验证码 + 持仓
产品主数据在 data/finance_kb/catalog.jsonl (唯一事实源, 由 core.slot_filler 加载后共享)。

【架构说明】以下账户/交易/信用卡/贷款/资产总览数据均为 Mock 演示数据,
生产环境应通过银行核心系统 API (账户系统/交易系统/信用卡系统/信贷系统/数据中台) 实时查询。
工具层已预留标准接口, 切换真实 API 时只需替换本文件中的数据获取逻辑, 上层 Agent 调用方式不变。
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

# ── 以下为银行核心系统对接预留接口的 Mock 数据 ──────────────────────────
# 生产环境替换为: 账户系统 API / 交易流水 API / 信用卡核心 API / 信贷系统 API / 数据中台

# 账户余额 (单位: 元)
ACCOUNT_BALANCES = {
    "u001": {
        "savings": {"account_no": "6222****1234", "balance": 28650.50, "type": "储蓄卡"},
        "fixed_deposits": [
            {"account_no": "0012****5678", "amount": 50000, "term": "1年", "rate": "1.95%", "maturity": "2027-03-15"},
            {"account_no": "0012****5679", "amount": 30000, "term": "3年", "rate": "2.35%", "maturity": "2028-06-20"},
        ],
        "total_deposit": 108650.50,
    },
    "u002": {
        "savings": {"account_no": "6222****5678", "balance": 15230.00, "type": "储蓄卡"},
        "fixed_deposits": [
            {"account_no": "0012****9012", "amount": 100000, "term": "2年", "rate": "2.15%", "maturity": "2027-11-10"},
        ],
        "total_deposit": 115230.00,
    },
    "u003": {
        "savings": {"account_no": "6222****9012", "balance": 5800.00, "type": "储蓄卡"},
        "fixed_deposits": [],
        "total_deposit": 5800.00,
    },
}

# 最近交易明细 (单位: 元, 按时间倒序)
TRANSACTIONS = {
    "u001": [
        {"date": "2026-09-25", "desc": "理财赎回-日日盈现金管理类", "amount": -10000, "balance_after": 28650.50, "type": "理财"},
        {"date": "2026-09-24", "desc": "工资入账", "amount": 15800, "balance_after": 38650.50, "type": "收入"},
        {"date": "2026-09-22", "desc": "理财申购-稳盈添利30天", "amount": -50000, "balance_after": 22850.50, "type": "理财"},
        {"date": "2026-09-20", "desc": "消费-超市", "amount": -356.80, "balance_after": 72850.50, "type": "消费"},
        {"date": "2026-09-18", "desc": "转账-他人汇入", "amount": 20000, "balance_after": 73207.30, "type": "转账"},
    ],
    "u002": [
        {"date": "2026-09-25", "desc": "理财收益到账-平衡增利180天", "amount": 420.50, "balance_after": 15230.00, "type": "理财收益"},
        {"date": "2026-09-20", "desc": "理财申购-平衡增利180天", "amount": -100000, "balance_after": 14809.50, "type": "理财"},
        {"date": "2026-09-15", "desc": "工资入账", "amount": 22000, "balance_after": 114809.50, "type": "收入"},
    ],
    "u003": [
        {"date": "2026-09-23", "desc": "消费-餐饮", "amount": -286.00, "balance_after": 5800.00, "type": "消费"},
        {"date": "2026-09-20", "desc": "转账-他人汇入", "amount": 3000, "balance_after": 6086.00, "type": "转账"},
    ],
}

# 信用卡信息
CREDIT_CARDS = {
    "u001": {
        "card_no": "6228****8888",
        "credit_limit": 50000,
        "available_limit": 32150,
        "current_bill": 17850,
        "min_repayment": 1785,
        "bill_date": "2026-09-10",
        "repayment_due": "2026-10-05",
        "status": "正常",
    },
    "u002": {
        "card_no": "6228****9999",
        "credit_limit": 80000,
        "available_limit": 65400,
        "current_bill": 14600,
        "min_repayment": 1460,
        "bill_date": "2026-09-15",
        "repayment_due": "2026-10-10",
        "status": "正常",
    },
    "u003": {},
}

# 贷款信息
LOANS = {
    "u001": {
        "loan_no": "LN2024****001",
        "type": "住房按揭贷款",
        "principal_remaining": 856000,
        "monthly_repayment": 5230.50,
        "next_repayment_date": "2026-10-01",
        "rate": "4.2%",
        "term_remaining": "23年6个月",
        "status": "正常",
    },
    "u002": {},
    "u003": {},
}

# 理财订单 (申购/赎回处理中)
ORDERS = {
    "u001": [
        {"order_id": "ORD20260925001", "product_name": "日日盈现金管理类", "type": "赎回",
         "amount": 10000, "status": "已到账", "submit_time": "2026-09-25 10:30", "complete_time": "2026-09-25 14:00"},
    ],
    "u002": [
        {"order_id": "ORD20260920001", "product_name": "平衡增利180天", "type": "申购",
         "amount": 100000, "status": "已确认", "submit_time": "2026-09-20 09:15", "complete_time": "2026-09-21 15:00"},
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


# ── 银行核心系统对接预留接口 (Mock 实现) ──────────────────────────────
# 生产环境: 替换为对账户系统/交易系统/信用卡系统/信贷系统/数据中台的 API 调用

def get_account_balance(user_id: str) -> dict:
    """查询账户余额 (储蓄卡 + 定期存款)
    生产环境对接: 银行核心账户系统 API"""
    return ACCOUNT_BALANCES.get(user_id, {})


def get_transactions(user_id: str, limit: int = 10) -> list:
    """查询最近交易明细
    生产环境对接: 交易流水系统 API"""
    return TRANSACTIONS.get(user_id, [])[:limit]


def get_credit_card(user_id: str) -> dict:
    """查询信用卡信息 (额度/账单/还款日)
    生产环境对接: 信用卡核心系统 API"""
    return CREDIT_CARDS.get(user_id, {})


def get_loan(user_id: str) -> dict:
    """查询贷款信息 (余额/月供/利率)
    生产环境对接: 信贷管理系统 API"""
    return LOANS.get(user_id, {})


def get_asset_overview(user_id: str) -> dict:
    """查询资产总览 (存款+理财+基金+保险-负债)
    生产环境对接: 数据中台/统一客户视图 API"""
    profile = PROFILES.get(user_id, {})
    account = ACCOUNT_BALANCES.get(user_id, {})
    holdings = HOLDINGS.get(user_id, [])
    loan = LOANS.get(user_id, {})

    total_investment = sum(h["amount_wan"] * 10000 for h in holdings)
    total_deposit = account.get("total_deposit", 0)
    total_assets = total_deposit + total_investment
    total_liabilities = loan.get("principal_remaining", 0)
    net_assets = total_assets - total_liabilities

    return {
        "user_name": profile.get("name", ""),
        "total_assets": total_assets,
        "total_deposit": total_deposit,
        "total_investment": total_investment,
        "total_liabilities": total_liabilities,
        "net_assets": net_assets,
        "asset_structure": {
            "活期及定期存款": total_deposit,
            "理财及基金": total_investment,
        },
    }


def get_orders(user_id: str) -> list:
    """查询理财订单状态 (申购/赎回)
    生产环境对接: 理财业务系统订单 API"""
    return ORDERS.get(user_id, [])
