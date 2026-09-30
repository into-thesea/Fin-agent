"""
P2: 监控指标聚合 (Redis 原子计数 + 内存降级)

指标: QPS / 成功率 / 错误率 / 超时率 / 缓存命中率 / 平均延迟 / P50-P90-P99
存储: Redis (前缀 metrics:, 按小时桶) 原子 INCR; Redis 不可用时降级进程内内存。
延迟: 每小时桶存一个 List (上限 MAX_LATENCY_SAMPLES), 读取时排序算分位数。

用法 (单例):
    from src.monitor.metrics_store import metrics_store
    metrics_store.record(status_code, duration_ms)   # 中间件每请求调用(非流式)
    metrics_store.record_stream(ttfb_ms, total_ms, ok)  # SSE 流结束/中断时调用
    metrics_store.incr_cache_hit()                   # 缓存命中处调用
    metrics_store.snapshot(window="1h")              # SLI 接口读取

流式请求单独成序列: 中间件测到的 SSE 耗时是「响应对象返回」的时间(~1ms), 不是用户感知的
首 token 时间, 混进全局分位数会把 p99 拉低成假象 —— 所以流式走 record_stream, 存
stream_ttfb / stream_total 两条序列, 不进全局 lat 序列(但仍计入 total / error)。
"""

from __future__ import annotations

import os
import time
import threading
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

METRICS_PREFIX = "metrics:"
MAX_LATENCY_SAMPLES = int(os.getenv("METRICS_MAX_LATENCY_SAMPLES", "20000"))  # 每桶延迟样本上限
METRICS_KEEP_BUCKETS = int(os.getenv("METRICS_KEEP_BUCKETS", "48"))          # 保留最近 N 个小时桶


def _percentile(sorted_lat: list, p: float) -> int:
    """有序延迟列表的 P 分位数 (毫秒)"""
    if not sorted_lat:
        return 0
    k = max(0, min(len(sorted_lat) - 1, int(len(sorted_lat) * p)))
    return int(sorted_lat[k])


class MetricsStore:
    _instance: Optional['MetricsStore'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'MetricsStore':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._mem: dict = {}       # {bucket: {total, cache_hit, error, timeout, latencies:[]}}
        self._mem_lock = threading.Lock()
        self._redis = None
        self._use_redis = False
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            from src.cache.redis_client import RedisCache
            cache = RedisCache()
            if cache.ping():
                self._redis = cache
                self._use_redis = True
                logger.info("MetricsStore: Redis 存储已启用")
                return
        except Exception as e:
            logger.debug("MetricsStore Redis 连接失败: %s", e)
        self._use_redis = False
        logger.info("MetricsStore: 降级进程内内存存储 (Redis 不可用)")

    # ── 内部: 分桶与自增 ──────────────────────────

    def _bucket(self, ts: Optional[float] = None) -> int:
        return int((ts or time.time()) // 3600)

    def _key(self, bucket: int, field: str) -> str:
        return f"{METRICS_PREFIX}{bucket}:{field}"

    def _incr(self, bucket: int, field: str, n: int = 1) -> None:
        if self._use_redis:
            try:
                key = self._key(bucket, field)
                self._redis.client.incrby(key, n)
                self._redis.client.expire(key, METRICS_KEEP_BUCKETS * 3600)
                return
            except Exception as e:
                logger.warning("MetricsStore Redis 写入失败, 降级内存: %s", e)
                self._use_redis = False
        with self._mem_lock:
            b = self._mem.setdefault(bucket, {
                "total": 0, "cache_hit": 0, "error": 0, "timeout": 0, "lat": [],
            })
            b[field] = b.get(field, 0) + n

    def _push_latency(self, bucket: int, ms: float, field: str = "lat") -> None:
        if self._use_redis:
            try:
                key = self._key(bucket, field)
                self._redis.client.rpush(key, int(ms))
                self._redis.client.ltrim(key, -MAX_LATENCY_SAMPLES, -1)
                self._redis.client.expire(key, METRICS_KEEP_BUCKETS * 3600)
                return
            except Exception as e:
                logger.warning("MetricsStore Redis 延迟写入失败, 降级内存: %s", e)
                self._use_redis = False
        with self._mem_lock:
            b = self._mem.setdefault(bucket, {
                "total": 0, "cache_hit": 0, "error": 0, "timeout": 0, "lat": [],
            })
            lst = b.setdefault(field, [])
            lst.append(int(ms))
            if len(lst) > MAX_LATENCY_SAMPLES:
                del lst[:-MAX_LATENCY_SAMPLES]

    # ── 对外记录接口 ──────────────────────────────

    def record(self, status_code: int, duration_ms: float) -> None:
        """中间件每请求调用(非流式): 计数 + 延迟采样"""
        bucket = self._bucket()
        self._incr(bucket, "total")
        if status_code == 504:
            self._incr(bucket, "timeout")
        elif status_code >= 500:
            self._incr(bucket, "error")
        self._push_latency(bucket, duration_ms)

    def record_stream(self, ttfb_ms: Optional[float], total_ms: float,
                      ok: bool = True) -> None:
        """流式请求结束时调用: 首 token 时间与总时长各自成序列。

        ttfb_ms 为 None 表示整条流没产出过答案(超时/异常), 此时只记总时长。
        流式请求仍计入 total(与 error), 因为它确实是请求 —— 进成功率的分子分母,
        只是它的耗时不能进全局延迟样本(见文件头说明)。
        """
        bucket = self._bucket()
        self._incr(bucket, "total")
        self._incr(bucket, "stream_cnt")
        if not ok:
            self._incr(bucket, "error")
            self._incr(bucket, "stream_error")
        if ttfb_ms is not None:
            self._push_latency(bucket, ttfb_ms, field="stream_ttfb")
        self._push_latency(bucket, total_ms, field="stream_total")

    def incr_cache_hit(self) -> None:
        """缓存命中处调用 (chat.py 命中语义缓存时)"""
        self._incr(self._bucket(), "cache_hit")

    # ── 读取接口 ──────────────────────────────────

    def snapshot(self, window: str = "1h") -> dict:
        """
        聚合快照。window: "1h"=当前小时桶 / "24h"=最近24个桶。
        p*/avg_ms 只统计**非流式**请求; 流式见 stream_* 一组(ttfb / total 各自分位)。
        Returns: {"window", "buckets", "total", "qps", "success_rate",
                  "error_rate", "timeout_rate", "cache_hit_rate",
                  "avg_ms", "p50_ms", "p90_ms", "p99_ms", "samples",
                  "stream_cnt", "stream_errors", "stream_samples",
                  "stream_ttfb_p50/p90/p99_ms", "stream_total_p50/p90/p99_ms"}
        """
        now = time.time()
        cur = self._bucket(now)
        buckets = [cur] if window == "1h" else list(range(cur - 23, cur + 1))

        total = cache_hit = error = timeout = 0
        stream_cnt = stream_error = 0
        latencies: list = []
        ttfb: list = []
        stream_total: list = []
        for b in buckets:
            if self._use_redis:
                try:
                    total += int(self._redis.client.get(self._key(b, "total")) or 0)
                    cache_hit += int(self._redis.client.get(self._key(b, "cache_hit")) or 0)
                    error += int(self._redis.client.get(self._key(b, "error")) or 0)
                    timeout += int(self._redis.client.get(self._key(b, "timeout")) or 0)
                    stream_cnt += int(self._redis.client.get(self._key(b, "stream_cnt")) or 0)
                    stream_error += int(self._redis.client.get(self._key(b, "stream_error")) or 0)
                    latencies += [int(x) for x in self._redis.client.lrange(self._key(b, "lat"), 0, -1)]
                    ttfb += [int(x) for x in self._redis.client.lrange(self._key(b, "stream_ttfb"), 0, -1)]
                    stream_total += [int(x) for x in self._redis.client.lrange(self._key(b, "stream_total"), 0, -1)]
                    continue
                except Exception as e:
                    logger.warning("MetricsStore Redis 读取失败, 降级内存: %s", e)
                    self._use_redis = False
            with self._mem_lock:
                bd = self._mem.get(b)
                if bd:
                    total += bd["total"]
                    cache_hit += bd["cache_hit"]
                    error += bd["error"]
                    timeout += bd["timeout"]
                    stream_cnt += bd.get("stream_cnt", 0)
                    stream_error += bd.get("stream_error", 0)
                    latencies += bd.get("lat", [])
                    ttfb += bd.get("stream_ttfb", [])
                    stream_total += bd.get("stream_total", [])

        latencies.sort()
        ttfb.sort()
        stream_total.sort()
        window_sec = 3600 if window == "1h" else min(24 * 3600, now - buckets[0] * 3600)
        window_sec = max(window_sec, 1)
        return {
            "window": window,
            "buckets": len(buckets),
            "total": total,
            "qps": round(total / window_sec, 3),
            "success_rate": round((total - error - timeout) / max(total, 1), 4),
            "error_rate": round(error / max(total, 1), 4),
            "timeout_rate": round(timeout / max(total, 1), 4),
            "cache_hit_rate": round(cache_hit / max(total, 1), 4),
            "avg_ms": round(sum(latencies) / max(len(latencies), 1), 1),
            "p50_ms": _percentile(latencies, 0.50),
            "p90_ms": _percentile(latencies, 0.90),
            "p99_ms": _percentile(latencies, 0.99),
            "samples": len(latencies),
            # 流式单独一组: 用户感知的是 TTFB, 总时长另有意义(见文件头说明)
            "stream_cnt": stream_cnt,
            "stream_errors": stream_error,
            "stream_samples": len(stream_total),
            "stream_ttfb_p50_ms": _percentile(ttfb, 0.50),
            "stream_ttfb_p90_ms": _percentile(ttfb, 0.90),
            "stream_ttfb_p99_ms": _percentile(ttfb, 0.99),
            "stream_total_p50_ms": _percentile(stream_total, 0.50),
            "stream_total_p90_ms": _percentile(stream_total, 0.90),
            "stream_total_p99_ms": _percentile(stream_total, 0.99),
        }


# 全局单例
metrics_store = MetricsStore()
