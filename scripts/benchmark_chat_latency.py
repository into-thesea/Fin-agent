"""
Fin-Agent 问答时延基准测试 — 端到端时延 + 流式 TTFB + 质量指标

测量:
  1. 同步接口 /api/v1/chat/sync 端到端时延 (p50/p90/p95/max)
  2. 流式接口 /api/v1/chat/stream 首token时延 (TTFB) + 总时延
  3. 每次回答的质量代理指标: sources 数量, review score, answer 长度, intent, 规则分类

用法:
    python scripts/benchmark_chat_latency.py            # 默认问题集
    python scripts/benchmark_chat_latency.py --query "..."  # 单条
    python scripts/benchmark_chat_latency.py --repeat 2 --out bench_results.json
"""

import sys
import os
import json
import time
import argparse
import statistics
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8001"

# 覆盖三类路由: complex(完整管道带检索) / simple_fact(快速通道不检索) / greeting(问候)
DEFAULT_QUERIES = [
    # 复杂 → 完整 Pipeline (检索 + 领域Agent + 审核)
    "比亚迪主要从事哪些业务？",
    "对比比亚迪和特斯拉2024年的经营表现",
    "小米集团近3年毛利率变化趋势",
    "比亚迪2024年为什么业绩增长这么快",
    # 简单事实 → 快速通道 (仅LLM, 不检索)
    "比亚迪2024年营收是多少？",
    "特斯拉2023年净利润是多少？",
    "茅台2024年分红是多少？",
    # 问候 → 极速
    "你好",
]


def sync_chat(query: str, timeout: float = 150) -> dict:
    """同步问答, 返回 (耗时, 响应)"""
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/api/v1/chat/sync", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "latency": time.time() - t0, "data": data}
    except Exception as e:
        return {"ok": False, "latency": time.time() - t0, "error": f"{type(e).__name__}: {str(e)[:100]}"}


def stream_chat(query: str, timeout: float = 150) -> dict:
    """流式问答, 测首token时延 (TTFB) 与总时延"""
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/api/v1/chat/stream", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttfb = None
    total_tokens = 0
    answer = ""
    sources = []
    review = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp.read().decode("utf-8").splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                try:
                    msg = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if ttfb is None and msg.get("type") in ("token", "result"):
                    ttfb = time.time() - t0
                if msg.get("type") == "token":
                    total_tokens += 1
                    answer += msg.get("token", "")
                elif msg.get("type") == "result":
                    answer = msg.get("answer", answer)
                    sources = msg.get("sources", [])
                    review = msg.get("review", {})
                elif msg.get("type") == "error":
                    return {"ok": False, "latency": time.time() - t0, "ttfb": ttfb,
                            "error": msg.get("message", "未知错误")}
        return {
            "ok": True, "latency": time.time() - t0, "ttfb": ttfb,
            "tokens": total_tokens, "answer_len": len(answer),
            "sources": sources, "review": review,
        }
    except Exception as e:
        return {"ok": False, "latency": time.time() - t0, "ttfb": ttfb,
                "error": f"{type(e).__name__}: {str(e)[:100]}"}


def main():
    ap = argparse.ArgumentParser(description="Fin-Agent 问答时延基准")
    ap.add_argument("--query", default="", help="单条查询 (默认跑内置问题集)")
    ap.add_argument("--repeat", type=int, default=1, help="每条重复次数")
    ap.add_argument("--stream", action="store_true", help="测流式 TTFB")
    ap.add_argument("--out", default="", help="保存结果 JSON")
    args = ap.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    print(f"📋 基准问题数: {len(queries)} (重复 {args.repeat} 次)  模式: {'流式' if args.stream else '同步'}\n")

    rows = []
    for q in queries:
        for i in range(args.repeat):
            tag = f"[{i+1}/{args.repeat}]" if args.repeat > 1 else ""
            if args.stream:
                r = stream_chat(q)
                rows.append({"query": q, **r})
                status = "OK" if r["ok"] else "FAIL"
                ttfb = r.get("ttfb")
                ttfb_str = f"{ttfb:.1f}s" if ttfb else "-"
                print(f"  {status} {tag} {q[:30]:<32} 总{r['latency']:6.1f}s 首token{ttfb_str:<8} "
                      f"tok={r.get('tokens', 0):<4} src={len(r.get('sources', []))}")
            else:
                r = sync_chat(q)
                rows.append({"query": q, **r})
                if r["ok"]:
                    d = r["data"]
                    print(f"  OK   {tag} {q[:30]:<32} {r['latency']:6.1f}s  src={len(d.get('sources', []))} "
                          f"intent={d.get('_intent')} rule={d.get('_rule_result')} "
                          f"ans={len(d.get('answer',''))}c review={d.get('review',{}).get('score')}")
                else:
                    print(f"  FAIL {tag} {q[:30]:<32} {r['latency']:6.1f}s  {r.get('error')}")

    # ── 汇总统计 ──────────────────────────────
    ok_rows = [r for r in rows if r["ok"]]
    if not ok_rows:
        print("\n❌ 全部失败")
        return
    lat = sorted(r["latency"] for r in ok_rows)
    n = len(lat)
    def pct(p): return lat[min(n - 1, int(n * p))]
    print("\n" + "=" * 46)
    print(f"📊 时延汇总 (成功 {n}/{len(rows)})")
    print("=" * 46)
    print(f"  mean   = {statistics.mean(lat):6.2f}s")
    print(f"  p50    = {pct(0.50):6.2f}s")
    print(f"  p90    = {pct(0.90):6.2f}s")
    print(f"  p95    = {pct(0.95):6.2f}s")
    print(f"  max    = {max(lat):6.2f}s")
    if args.stream:
        ttfb_list = [r["ttfb"] for r in ok_rows if r.get("ttfb")]
        if ttfb_list:
            print(f"\n  首token(TTFB): mean={statistics.mean(ttfb_list):.2f}s "
                  f"p50={sorted(ttfb_list)[len(ttfb_list)//2]:.2f}s")
        toks = [r.get("tokens", 0) for r in ok_rows if r.get("tokens")]
        if toks:
            print(f"  平均 token 数 = {statistics.mean(toks):.0f}  "
                  f"token/s = {statistics.mean(toks) / max(statistics.mean(lat), 0.001):.1f}")
    print("=" * 46)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"mode": "stream" if args.stream else "sync", "rows": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"✅ 逐条结果已保存: {args.out}")


if __name__ == "__main__":
    main()
