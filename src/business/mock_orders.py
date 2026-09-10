"""
Mock 订单数据源 (Phase 2 演示用)

示例订单数据，后续接入真实业务 API (BUSINESS_API_BASE_URL) 时替换本模块。
订单字段: order_id, user_id, items, status, carrier, tracking_no, total, ship_fee,
          shipped_at, received_days_ago, return_window_days
"""

MOCK_ORDERS = {
    "SO100001": {
        "order_id": "SO100001",
        "user_id": "u001",
        "items": [{"name": "B3无线蓝牙耳机", "qty": 1, "price": 199.0}],
        "status": "已发货",
        "carrier": "顺丰",
        "tracking_no": "SF1234567890",
        "total": 199.0,
        "ship_fee": 0.0,
        "shipped_at": "2026-08-20 10:30",
        "received_days_ago": 0,
        "return_window_days": 7,
    },
    "SO100002": {
        "order_id": "SO100002",
        "user_id": "u001",
        "items": [{"name": "家居收纳箱", "qty": 2, "price": 59.0}],
        "status": "已签收",
        "carrier": "京东",
        "tracking_no": "JD9876543210",
        "total": 118.0,
        "ship_fee": 8.0,
        "shipped_at": "2026-08-12 14:20",
        "received_days_ago": 8,  # 已签收 8 天, 超过 7 天退换期 (用于验证 handoff)
        "return_window_days": 7,
    },
    "SO200001": {
        "order_id": "SO200001",
        "user_id": "u002",
        "items": [{"name": "无线鼠标", "qty": 1, "price": 89.0}],
        "status": "待发货",
        "carrier": "普通快递",
        "tracking_no": "",
        "total": 89.0,
        "ship_fee": 8.0,
        "shipped_at": "",
        "received_days_ago": 0,
        "return_window_days": 7,
    },
}
