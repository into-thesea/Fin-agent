"""
业务工具服务 — 订单/物流/运费/退款

统一返回结构: {"text": str(发给LLM), "sources": [], "contexts": {...}, "handoff": bool}
与 src/tools/registry.py 的 EXECUTE_TOOLS 约定一致。
订单归属校验: order.user_id == 当前用户 (get_current_user), 无当前用户时不强制。
"""

from .context import get_current_user
from .mock_orders import MOCK_ORDERS

# 偏远地区运费标记
_REMOTE = ["新疆", "西藏", "内蒙古"]
_MAINLAND = ["上海", "北京", "广东", "江苏", "浙江"]


def _ok(text, handoff=False):
    return {
        "text": text,
        "sources": [],
        "contexts": {"local": [], "global": [], "graph": []},
        "handoff": handoff,
    }


def _owned(order_id: str):
    """返回 (order, 拒绝信息) — 校验订单存在 + 归属"""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return None, f"未找到订单 {order_id}，请核对订单号。"
    uid = get_current_user()
    if uid and order["user_id"] != uid:
        return None, f"订单 {order_id} 不属于当前用户，无权查看他人订单。"
    return order, None


def query_order(order_id: str) -> dict:
    """查询订单状态与详情"""
    order, denied = _owned(order_id)
    if denied:
        return _ok(f"【订单查询】{denied}")
    items = "、".join(f"{i['name']}x{i['qty']}" for i in order["items"])
    text = (f"【订单查询】订单 {order['order_id']} 当前状态：{order['status']}。\n"
            f"商品：{items}\n金额：{order['total']:.0f}元（运费 {order['ship_fee']:.0f}元）\n"
            f"承运：{order['carrier']}，运单号：{order['tracking_no'] or '尚未发货'}")
    return _ok(text)


def query_shipping(order_id: str) -> dict:
    """查询物流进度"""
    order, denied = _owned(order_id)
    if denied:
        return _ok(f"【物流查询】{denied}")
    if not order["tracking_no"]:
        return _ok(f"【物流查询】订单 {order['order_id']} 尚未发货，暂无物流信息。")
    text = (f"【物流查询】订单 {order['order_id']} 由{order['carrier']}承运，运单号 {order['tracking_no']}。\n"
            f"当前状态：{order['status']}，发货时间：{order['shipped_at']}。")
    return _ok(text)


def estimate_shipping(address: str, items: str = "") -> dict:
    """按收货地址估算运费 (简化规则)"""
    if any(k in address for k in _REMOTE):
        fee, note = 25, "偏远地区，运费以结算页为准"
    elif any(k in address for k in _MAINLAND):
        fee, note = 15, "京东快递首重"
    else:
        fee, note = 10, "普通快递首重"
    text = f"【运费估算】寄往「{address}」预估运费 {fee} 元（{note}，超重续费另计）。"
    return _ok(text)


def check_refund_eligibility(order_id: str) -> dict:
    """只读: 校验退款资格 (无副作用, P3 图确定性门控用)

    Returns: {"eligible": bool, "handoff": bool, "reason": str}
      - eligible=True  → 在退换期内, 可在线退款
      - handoff=True   → 超期, 需转人工
    """
    order, denied = _owned(order_id)
    if denied:
        return {"eligible": False, "handoff": False, "reason": denied}
    if order["received_days_ago"] > order["return_window_days"]:
        return {
            "eligible": False, "handoff": True,
            "reason": f"订单 {order_id} 已签收 {order['received_days_ago']} 天，"
                      f"超过 {order['return_window_days']} 天退换期，无法在线申请退款，建议转人工客服处理。",
        }
    return {"eligible": True, "handoff": False, "reason": ""}


def submit_refund(order_id: str, reason: str = "") -> dict:
    """提交退款申请; 超退换期 → handoff=True"""
    order, denied = _owned(order_id)
    if denied:
        return _ok(f"【退款申请】{denied}")
    if order["received_days_ago"] > order["return_window_days"]:
        return _ok(
            f"【退款申请】订单 {order['order_id']} 已签收 {order['received_days_ago']} 天，"
            f"超过 {order['return_window_days']} 天退换期，无法在线申请退款，建议转人工客服处理。",
            handoff=True,
        )
    text = (f"【退款申请】订单 {order['order_id']} 在退换期内，已为您记录退款申请（原因：{reason or '未填写'}）。\n"
            f"请按退货流程寄回商品，验收后 1-3 个工作日原路退款。")
    return _ok(text)
