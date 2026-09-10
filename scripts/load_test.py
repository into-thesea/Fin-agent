"""
P6: 并发压测脚本 — 量化 QPS / P50-P95-P99 延迟 / 错误率 / 缓存命中率

用法:
  .venv/Scripts/python.exe scripts/load_test.py \
      --url http://127.0.0.1:8001/api/v1/chat/stream \
      --queries "顺丰快递首重是多少" "全场默认用哪种快递" \
      --concurrency 20 --requests 60

输出: QPS / P50 / P90 / P95 / P99 (ms) / 平均 / 成功率 / 缓存命中率
缓存命中判定 (stream): 响应无 token 事件且含 result → 命中。
"""

from __future__ import annotations

import sys
import os
import time
import asyncio
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx


def percentile(lat_ms: list, p: float) -> int:
    if not lat_ms:
        return 0
    s = sorted(lat_ms)
    return int(s[min(len(s) - 1, int(len(s) * p))])


async def worker(client: httpx.AsyncClient, url: str, queries: list,
                 results: list, sem: asyncio.Semaphore):
    while True:
        async with sem:
            q = queries.pop(0) if queries else None
        if q is None:
            return
        t0 = time.time()
        try:
            async with client.stream("POST", url, json={"query": q, "session_id": "loadtest"}, timeout=240) as resp:
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                dt = (time.time() - t0) * 1000
                text = body.decode("utf-8", errors="replace")
                ok = resp.status_code == 200 and '"type": "result"' in text
                cached = ok and '"type": "token"' not in text
                results.append({"lat": dt, "ok": ok, "cached": cached, "q": q[:20], "status": resp.status_code})
        except Exception as e:
            dt = (time.time() - t0) * 1000
            results.append({"lat": dt, "ok": False, "cached": False, "q": q[:20], "status": "err:" + str(e)[:30]})


async def run(url: str, queries: list, concurrency: int, requests: int):
    # 循环取问题直到达到 requests
    pool = []
    for _ in range(requests):
        pool.append(queries[len(pool) % len(queries)])
    results: list = []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [worker(client, url, pool, results, sem) for _ in range(concurrency)]
        await asyncio.gather(*tasks)

    lats = [r["lat"] for r in results]
    ok = [r for r in results if r["ok"]]
    cached = [r for r in ok if r["cached"]]
    n = len(results)
    elapsed = max(lats) / 1000 if lats else 0

    print("\n" + "=" * 50)
    print("📊 压测报告")
    print("=" * 50)
    print(f"  目标: {url}")
    print(f"  并发: {concurrency} | 请求数: {n} | 耗时: {elapsed:.1f}s")
    print(f"  QPS         {n / max(elapsed, 0.001):.2f}")
    print(f"  成功率      {len(ok) / max(n, 1):.1%}")
    print(f"  缓存命中率  {len(cached) / max(len(ok), 1):.1%}")
    print(f"  P50         {percentile(lats, 0.50)} ms")
    print(f"  P90         {percentile(lats, 0.90)} ms")
    print(f"  P95         {percentile(lats, 0.95)} ms")
    print(f"  P99         {percentile(lats, 0.99)} ms")
    print(f"  平均        {statistics.mean(lats):.0f} ms")
    print("=" * 50)
    # 失败样例
    fails = [r for r in results if not r["ok"]][:3]
    for f in fails:
        print(f"  ⚠️ 失败: {f['q']} -> {f['status']} ({f['lat']:.0f}ms)")


def main():
    parser = argparse.ArgumentParser(description="并发压测")
    parser.add_argument("--url", default="http://127.0.0.1:8001/api/v1/chat/stream")
    parser.add_argument("--queries", nargs="+", default=["顺丰快递首重是多少"])
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.queries, args.concurrency, args.requests))


if __name__ == "__main__":
    main()
