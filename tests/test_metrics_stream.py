"""流式请求的可信延迟打点 —— TTFB / 总时长独立成序列, 不混进全局 HTTP 分位数。

为什么需要:
  1) SSE 走中间件测出来的 latency 是「响应对象返回」的耗时(实测 ~1ms), 不是用户感知的
     首 token 时间; 它混进全局分位数会把 p99 拉成假象(实测流式 230 个样本 p99=71ms,
     而基准脚本测出的真实 TTFB 是 27.6s)。
  2) 所以流式请求必须: ① 不进全局延迟样本; ② 自己的首 token 时间与总时长单独成序列。

覆盖四件事:
  1. record_stream 把 TTFB / 总时长分开存, 且不污染全局 HTTP 分位数
  2. 流式失败(超时/异常)计错误, 且没有 TTFB 时只进总时长序列
  3. 中间件识别 SSE 并按 content-type 跳过全局延迟采样(不能用 isinstance——
     BaseHTTPMiddleware 的 call_next 永远返回 _StreamingResponse(Response 子类))
  4. 包装器在首个 token 记 TTFB、流结束记总时长/成败, 且不吞异常
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest

from src.monitor.metrics_store import metrics_store


@pytest.fixture(autouse=True)
def clean_store():
    """单例: 用例前后清空内存存储并强制内存模式, 避免受 Redis 与环境污染。"""
    metrics_store._mem.clear()
    _use_redis = metrics_store._use_redis
    metrics_store._use_redis = False
    yield
    metrics_store._use_redis = _use_redis
    metrics_store._mem.clear()


def test_record_stream_keeps_ttfb_and_total_separate():
    for i in range(100):
        metrics_store.record_stream(ttfb_ms=100 + i, total_ms=20000 + i, ok=True)

    snap = metrics_store.snapshot("1h")
    assert snap["stream_samples"] == 100
    assert 190 <= snap["stream_ttfb_p99_ms"] <= 200
    assert 20000 <= snap["stream_total_p99_ms"] <= 20100
    # 关键: 流式样本不得污染全局 HTTP 分位数(非流式样本数为 0)
    assert snap["samples"] == 0
    assert snap["p99_ms"] == 0


def test_stream_failure_counted_and_no_ttfb_is_allowed():
    metrics_store.record_stream(ttfb_ms=None, total_ms=120000, ok=False)

    snap = metrics_store.snapshot("1h")
    assert snap["stream_errors"] == 1
    assert snap["stream_samples"] == 1      # 无 TTFB 的样本仍然记总时长
    assert snap["error_rate"] == pytest.approx(1.0)   # 流式请求也是请求, 要进成功率


def test_middleware_skips_streaming_response():
    """SSE 的中间件耗时不进全局延迟样本, 但 trace_id 回写照旧。"""
    from starlette.requests import Request

    from src.api.main import add_trace_id

    scope = {"type": "http", "method": "POST", "path": "/api/v1/chat/stream",
             "headers": []}

    async def call_next(_request):
        from fastapi.responses import StreamingResponse
        return StreamingResponse(iter(["data: {}\n\n"]),
                                 media_type="text/event-stream")

    resp = asyncio.run(add_trace_id(Request(scope), call_next))
    assert resp.headers["X-Trace-ID"]
    assert metrics_store.snapshot("1h")["samples"] == 0

    async def call_next_json(_request):
        from fastapi.responses import JSONResponse
        return JSONResponse({"ok": True})

    asyncio.run(add_trace_id(Request(scope), call_next_json))
    assert metrics_store.snapshot("1h")["samples"] == 1     # 非流式照常采样


def test_timed_stream_records_ttfb_total_and_reraises():
    from src.api.routes.chat import _timed_stream

    # t0 用真实起点: 日志是共享的, 假 t0 会把 epoch 毫秒写进真实日志里(见过)
    holder = {"t0": time.time() - 0.05, "ttfb": None}

    async def fake_stream():
        holder["ttfb"] = 12.0          # 模拟首个 token 到达
        yield "data: {}\n\n"
        yield "data: {}\n\n"

    async def drain(gen):
        return [c async for c in gen]

    chunks = asyncio.run(drain(_timed_stream(fake_stream(), holder)))
    assert len(chunks) == 2
    snap = metrics_store.snapshot("1h")
    assert snap["stream_samples"] == 1
    assert snap["stream_ttfb_p99_ms"] == 12
    assert snap["stream_errors"] == 0

    async def boom():
        raise RuntimeError("上游炸了")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError):
        asyncio.run(drain(_timed_stream(boom(), holder)))
    snap2 = metrics_store.snapshot("1h")
    assert snap2["stream_errors"] == 1
    assert snap2["stream_samples"] == 2
