"""缓存动态数据拒写测试 — 订单/物流等实时数据不进语义缓存"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.core.answer_cache import _is_error_response


def test_dynamic_intent_rejected():
    """订单/物流/退款/投诉 → 视为不可缓存"""
    assert _is_error_response({"_intent": "order"}) is True
    assert _is_error_response({"_intent": "shipping"}) is True
    assert _is_error_response({"_intent": "refund"}) is True
    assert _is_error_response({"_intent": "complaint"}) is True


def test_static_intent_allowed():
    """FAQ/问候等静态答案可缓存"""
    assert _is_error_response({"_intent": "faq", "answer": "退货政策是..."}) is False
    assert _is_error_response({"_intent": "greeting", "answer": "您好"}) is False


def test_error_intent_still_rejected():
    assert _is_error_response({"_intent": "error"}) is True
