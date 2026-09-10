"""
业务上下文 — 线程级当前用户

用于订单/物流等业务工具的归属校验 (order.user_id == 当前用户)。
在 WorkerPool 工作线程 / 流式分析线程内设置，工具函数通过 get_current_user() 读取。
"""

import threading

_local = threading.local()


def set_current_user(user_id):
    """设置当前线程的用户 ID (每个请求进入分析前调用)"""
    _local.current_user = user_id


def get_current_user():
    """读取当前线程的用户 ID (可能为 None)"""
    return getattr(_local, "current_user", None)
