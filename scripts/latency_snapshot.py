"""延迟分位数快照 —— 从结构化日志离线算 P50/P90/P95/P99, **按端点分组**。

为什么要按端点分组: 全局混算等于把 /health(2ms) 和 /chat/sync(76s) 平均在一起, 算出来的
"p99" 没有意义(实测: 全局 p99=12.8s, 而同步问答自己就是 75.9s)。报延迟必须带端点。

流式怎么算: SSE 走中间件测出来的耗时只到「响应对象返回」(实测 ~1ms), 不是用户感知的首字
时间。所以本脚本只认带 ttfb_ms 的流式日志行(修复后才产生), 更早的 SSE 行单独计数并标注忽略。

分位数算法直接复用 src.monitor.metrics_store._percentile —— 保证离线复盘与在线看板口径一致。

用法:
    python scripts/latency_snapshot.py                      # 全部端点
    python scripts/latency_snapshot.py --since 2026-09-01
    python scripts/latency_snapshot.py --path chat/sync --min-n 1

注意 --path 用**子串**匹配, 建议写成不带前导斜杠的形式(如 chat/sync): Git Bash 会把
以 / 开头的参数当成路径转换成 Windows 路径, 那样就匹配不到了。
"""
import argparse
import collections
import io
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 只是借用分位算法, 不需要存储 —— 压掉 metric store 启动时探测 Redis 的告警
logging.getLogger("src.cache.redis_client").setLevel(logging.CRITICAL)
logging.getLogger("src.monitor.metrics_store").setLevel(logging.CRITICAL)

from src.monitor.metrics_store import _percentile  # 同一套分位口径

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "logs", "fin-agent.log")
_MSG_RE = re.compile(r"(\w+) (\S+) -> (\d+)")
_MAX_PLAUSIBLE_MS = 24 * 3600 * 1000   # 超过一天的样本不可能是真实延迟(时钟/测量错误)


def _fmt(ms) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms}ms"


def load(log_path: str, since: str = "", path_filter: list = None,
         only_ok: bool = False) -> dict:
    """扫日志: 分 HTTP 行 / 流式行(带 ttfb_ms) / 修复前的 SSE 行(忽略)。

    only_ok=True 时只看 2xx: 4xx/5xx 有自己的耗时画像(如 400 是 0ms 直接拒),
    而且测试套件会用 TestClient 打真实日志, 混进来会把 p50 拉成假象。
    """
    http: dict = collections.defaultdict(list)
    stream_ttfb: dict = collections.defaultdict(list)
    stream_total: dict = collections.defaultdict(list)
    ignored_sse = 0
    suspect = 0
    with io.open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "latency_ms" not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if since and str(rec.get("timestamp", "")) < since:
                continue
            m = _MSG_RE.match(rec.get("message", ""))
            if not m:
                continue
            path = rec.get("path") or m.group(2)
            # 子串匹配: Git Bash 会把以 / 开头的参数转换成 Windows 路径, 所以支持
            # 写 `--path chat/sync` 这种不带前导斜杠的形式(见文件头用法)
            if path_filter and not any(f in path for f in path_filter):
                continue
            ms = int(rec.get("latency_ms") or 0)
            if ms > _MAX_PLAUSIBLE_MS:
                suspect += 1            # 时钟/测量错误(如假 t0 算出的 epoch 毫秒)
                continue
            if only_ok and not 200 <= int(rec.get("status") or m.group(3)) < 300:
                continue
            if rec.get("ttfb_ms") is not None or rec.get("stream"):
                stream_ttfb[path].append(int(rec["ttfb_ms"] or 0))
                stream_total[path].append(ms)
            elif path.endswith("/stream"):
                ignored_sse += 1        # 中间件测的 SSE 耗时(到响应对象返回)不可用
            else:
                http[path].append(ms)
    return {"http": http, "ttfb": stream_ttfb, "total": stream_total,
            "ignored_sse": ignored_sse, "suspect": suspect}


def report(data: dict, min_n: int = 5) -> None:
    rows = [(p, sorted(v)) for p, v in data["http"].items() if len(v) >= min_n]
    rows.sort(key=lambda x: -len(x[1]))
    print(f"{'端点':44} {'n':>6} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print("-" * 96)
    for path, v in rows:
        print(f"{path:44} {len(v):>6} {_fmt(_percentile(v, .5)):>8} "
              f"{_fmt(_percentile(v, .9)):>8} {_fmt(_percentile(v, .95)):>8} "
              f"{_fmt(_percentile(v, .99)):>8} {_fmt(v[-1]):>8}")

    if data["ttfb"]:
        print("\n流式 (TTFB / 总时长):")
        for path, tf in data["ttfb"].items():
            tf, tt = sorted(tf), sorted(data["total"][path])
            n = len(tf)
            print(f"{path:44} {n:>6} TTFB p50={_fmt(_percentile(tf, .5))} "
                  f"p90={_fmt(_percentile(tf, .9))} p99={_fmt(_percentile(tf, .99))} | "
                  f"总时长 p50={_fmt(_percentile(tt, .5))} p99={_fmt(_percentile(tt, .99))}")
    else:
        print("\n流式: 暂无带 TTFB 的样本(修复前只有中间件那版 ~1ms 的读数)")
    if data["ignored_sse"]:
        print(f"已忽略 {data['ignored_sse']} 条修复前的 SSE 记录"
              f"(中间件耗时=到响应对象返回, 不代表 TTFB)")
    if data["suspect"]:
        print(f"已忽略 {data['suspect']} 条不合理样本(> {_MAX_PLAUSIBLE_MS // 3600000}h, "
              f"时钟或测量错误)")

    total = sum(len(v) for v in data["http"].values())
    print(f"\n提示: 报延迟必须带端点与样本量; p99 在 n<1000 时只是「最慢的那几个请求」。"
          f"本次非流式样本 {total} 条。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--since", default="", help="ISO 前缀, 如 2026-09-01")
    ap.add_argument("--path", action="append", default=None)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--only-ok", action="store_true",
                    help="只统计 2xx(排除 4xx/5xx 与测试套件打的合成请求)")
    a = ap.parse_args()
    if not os.path.exists(a.log):
        sys.exit(f"日志不存在: {a.log}")
    report(load(a.log, a.since, a.path, a.only_ok), a.min_n)


if __name__ == "__main__":
    main()
