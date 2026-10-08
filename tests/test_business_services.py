"""业务工具服务测试 — 订单归属校验 / 物流 / 运费 / 退款转人工"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest

from src.business import context
from src.business.services import estimate_shipping, query_order, query_shipping, submit_refund


@pytest.fixture(autouse=True)
def _clear_user():
    yield
    context.set_current_user(None)


def test_query_order_owned():
    context.set_current_user("u001")
    r = query_order("SO100001")
    assert "已发货" in r["text"]
    assert r["handoff"] is False


def test_query_order_not_owned():
    context.set_current_user("u002")
    r = query_order("SO100001")
    assert "无权查看" in r["text"]


def test_query_order_no_user_not_enforced():
    context.set_current_user(None)
    r = query_order("SO100001")
    assert "已发货" in r["text"]  # 无当前用户时不强制校验


def test_query_order_not_found():
    context.set_current_user("u001")
    r = query_order("SO999999")
    assert "未找到订单" in r["text"]


def test_query_shipping():
    context.set_current_user("u001")
    r = query_shipping("SO100001")
    assert "SF1234567890" in r["text"]


def test_estimate_shipping_by_region():
    r = estimate_shipping("上海市浦东新区")
    assert "15" in r["text"]
    r2 = estimate_shipping("新疆乌鲁木齐")
    assert "25" in r2["text"]


def test_submit_refund_in_window():
    context.set_current_user("u001")
    r = submit_refund("SO100001", reason="不想要了")
    assert r["handoff"] is False
    assert "退款申请" in r["text"]


def test_submit_refund_beyond_window_triggers_handoff():
    context.set_current_user("u001")
    r = submit_refund("SO100002")  # 已签收 8 天 > 7 天退换期
    assert r["handoff"] is True
    assert "人工" in r["text"]
