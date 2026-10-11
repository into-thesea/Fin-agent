"""任务失败不能把 Celery worker 带走

回归点: `ProgressReporter.fail()` 曾经把普通 dict 当作 **FAILURE 状态**的 result
写进结果后端。Celery 约定 FAILURE 必须携带异常信息 (exc_type/exc_message/exc_module),
所以下一次对同一个 task_id 的 store 会走到
`_store_result → decode_result → meta_from_decoded → exception_to_python`,
在缺少 exc_type 时抛 `ValueError: Exception information must include the exception type`,
被 Celery 判为 `CRITICAL Unrecoverable error` —— **整个 worker 退出**。
更糟的是失败的消息会被 restore 回队列, 新 worker 捡起来再炸一次, 形成死循环。

实测: 2026-10-11 一次任务失败后 worker 11 秒内退出, 日志里就是这个 CRITICAL 报错。
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks import etl_tasks


class _FakeCache:
    def __init__(self):
        self.calls = []

    def set_task_status(self, **kwargs):
        self.calls.append(kwargs)


class _FakeTask:
    def __init__(self):
        self.request = SimpleNamespace(id="deadbeefcafe")
        self.calls = []

    def update_state(self, state=None, meta=None, **kwargs):
        self.calls.append((state, meta))


@pytest.fixture
def reporter(monkeypatch):
    cache = _FakeCache()
    monkeypatch.setattr(etl_tasks, "RedisCache", lambda: cache)
    task = _FakeTask()
    r = etl_tasks.ProgressReporter(task, total_steps=4)
    monkeypatch.setattr(r, "_sync_db", lambda *a, **k: None)
    return r


def test_fail_does_not_write_failure_state_to_celery(reporter):
    reporter.fail("boom")

    # 失败必须照旧上报 —— Redis + task_logs 才是前端真正读的两路
    assert reporter.cache.calls, "失败状态没有写进 Redis, 前端会一直显示进行中"
    assert reporter.cache.calls[-1]["status"] == "failed"
    assert reporter.cache.calls[-1]["error"] == "boom"

    # 但绝不能以 FAILURE 状态写普通 dict 进结果后端 —— 那会毒化这个 task_id
    poisoned = [(s, m) for s, m in reporter.task.calls if s == "FAILURE"]
    assert not poisoned, (
        "FAILURE 的 result 必须是异常信息; 写普通 dict 会让下次 store 这个 task_id "
        f"直接 CRITICAL 退出 worker。实际写了: {poisoned}"
    )


def test_report_still_publishes_custom_state(reporter):
    """进度上报照旧 (自定义状态 PROCESSING 不是异常状态, decode 不会碰它)"""
    reporter.report("vectorizing", step=1)
    assert [s for s, _ in reporter.task.calls] == ["PROCESSING"]
    assert reporter.cache.calls[-1]["status"] == "processing"


def test_complete_still_publishes_success(reporter):
    reporter.complete(doc_id=7, duration=1.5)
    assert [s for s, _ in reporter.task.calls] == ["SUCCESS"]
    assert reporter.cache.calls[-1]["status"] == "completed"


def test_processing_state_maps_to_processing_not_failed():
    """report() 发的 PROCESSING 必须映射成 processing

    未知状态会被 _normalize_state 判成 failed —— 正在跑的任务在 Celery 兜底那一路
    显示成「失败」, 比直接报错更难排查。
    """
    from src.api.routes.tasks import _normalize_state
    assert _normalize_state("PROCESSING") == "processing"
