"""
P6: 并发控制 — 限制 LLM 密集处理并发 (防线程爆炸 / 打爆 LLM API)

问题: /stream 每请求起一个后台线程, 无上限 → 高并发下线程爆炸, 且同时打爆 LLM API。
方案: processing_slot() 门控 (BoundedSemaphore), 默认 PROCESSING_CONCURRENCY=6,
      超出的请求排队最多 180s, 超时给降级响应 (不拒绝也不无限等)。

水平扩展前提: 已无状态化 (dialog_state 内存+Redis, 向量在 Milvus, 缓存前置) →
             多 worker/多实例时此门控按进程独立, 是受控吃满单机的合理边界。
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

PROCESSING_CONCURRENCY = int(os.getenv("PROCESSING_CONCURRENCY", "6"))
PROCESSING_QUEUE_TIMEOUT = int(os.getenv("PROCESSING_QUEUE_TIMEOUT", "180"))

_gate = threading.BoundedSemaphore(PROCESSING_CONCURRENCY)


@contextmanager
def processing_slot():
    """获取一个处理槽位 (LLM 密集任务用). 排队超时抛 TimeoutError。"""
    acquired = _gate.acquire(timeout=PROCESSING_QUEUE_TIMEOUT)
    try:
        if not acquired:
            raise TimeoutError(
                f"处理并发已达上限 ({PROCESSING_CONCURRENCY}), 排队超过 {PROCESSING_QUEUE_TIMEOUT}s, 请稍后重试"
            )
        yield
    finally:
        if acquired:
            _gate.release()


def gate_size() -> int:
    """当前可用槽位 (诊断用)"""
    return PROCESSING_CONCURRENCY - _gate._value if hasattr(_gate, "_value") else PROCESSING_CONCURRENCY
