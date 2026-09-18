"""
合成用户压测 + badcase 冷启动

模拟 N 个虚拟用户跑混合场景 (提问 / 多轮追问 / 超纲 / 投诉 / 举报 / 适当性拦截 /
申购 / 点踩), 一轮跑完同时产出两类指标:

  性能指标   QPS / P50-P90-P95-P99 / 成功率 / 错误率 / 超时率 / 缓存命中率
  运维指标   转人工量率 / 工单状态分布 / 工单结单时长 / badcase 候选池与原因分布
             / 分诊状态分布 / badcase 密度

QPS 用整轮墙钟时间算 —— 不是 max(单请求延迟), 后者在并发下会高估数倍。

用法:
  .venv/Scripts/python.exe scripts/synthetic_users.py --users 12 --rounds 3
  ... --no-agent          不模拟坐席接单 (工单停在 open)
  ... --downvote 0.2      20% 的轮次打点踩
  ... --promote 3         提升 3 条 badcase 为 Golden (会写 data/ 文件, 需显式开启)
  ... --report-only       只读库出报告, 不打流量
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx

API = os.getenv("API_BASE", "http://127.0.0.1:8001")
CHAT = f"{API}/api/v1/chat/stream"

# 演示客户只有 3 个 (src/business/mock_finance.py PROFILES), 虚拟用户按轮次复用这三个身份。
# 验证码也从 PROFILES 取, 不重打一遍 —— 改一处即可。
from src.business.mock_finance import PROFILES

USERS = list(PROFILES.keys())

# ── 场景池: 每个场景一个权重 + 若干"会话"(外层列表=一个用户的一次会话, 内层=多轮) ──
SCENARIOS = {
    "normal": {"w": 28, "convos": [
        ["稳盈添利30天业绩比较基准是多少"],
        ["存款保险保不保理财产品"],
        ["有哪些 R2 的理财"],
        ["理财产品赎回到账要几天"],
        ["申购费和赎回费怎么收"],
        ["银行理财是存款吗"],
        ["七日年化和万份收益有什么区别"],
    ]},
    "multi": {"w": 14, "convos": [
        ["我想买理财", "稳盈添利30天", "5万"],
        ["有哪些R2产品", "那第一个的期限多久"],
        ["这只产品风险大吗", "和我现在持有的比呢"],
    ]},
    "oos": {"w": 18, "convos": [     # 知识库外 / 超纲 → 制造 empty_retrieval / unknown
        ["你们银行的股价今天多少"],
        ["帮我推荐一只科技股"],
        ["今天的天气怎么样"],
        ["给我写一首诗"],
        ["隔壁银行的产品怎么样"],
    ]},
    "complaint": {"w": 12, "convos": [   # 确定性转人工 (HANDOFF_INTENTS)
        ["我要投诉你们的服务"],
        ["太差了，我要投诉"],
    ]},
    "fraud": {"w": 12, "convos": [
        ["有人打电话说能帮我操作理财，是不是诈骗"],
        ["我要举报飞单"],
    ]},
    "suitability": {"w": 12, "convos": [  # u001=R2 买 R4 → 门控拦截转人工
        ["我要申购进取增利365天 10万"],
        ["我要申购均衡配置固收增强 20万"],
    ]},
    "subscribe": {"w": 4, "convos": [     # 全流程 (验证码写死在 PROFILES)
        ["我要申购稳盈添利30天 3万"],
        ["我要申购日日盈现金管理类 1万"],
    ]},
}
_TOTAL_W = sum(s["w"] for s in SCENARIOS.values())


def _pick():
    r = random.uniform(0, _TOTAL_W)
    acc = 0.0
    for name, s in SCENARIOS.items():
        acc += s["w"]
        if r <= acc:
            return name, random.choice(s["convos"])
    return "normal", SCENARIOS["normal"]["convos"][0]


# ────────────────────────────── 虚拟用户 ──────────────────────────────

async def _turn(client, session_id, user_id, query, timeout):
    """跑一轮对话, 返回 (记录, trace_id)"""
    t0 = time.time()
    try:
        async with client.stream("POST", CHAT,
                                 json={"query": query, "session_id": session_id, "user_id": user_id},
                                 timeout=timeout) as resp:
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk
            dt = (time.time() - t0) * 1000
            text = body.decode("utf-8", errors="replace")
            ok = resp.status_code == 200 and '"type": "result"' in text
            rec = {
                "lat": dt, "ok": ok,
                # 注意: "无 token 事件" ≠ 缓存命中。静态节点 (handoff/subscribe) 直接返回
                # 预置话术, 也不发 token —— 早先把它当缓存命中算出过 92% 的假值。
                # 真正的缓存命中率以服务端 /monitor/sli 为准。
                "no_token": ok and '"type": "token"' not in text,
                "handoff": '"type": "handoff"' in text,
                "timeout": False,
                "status": resp.status_code,
                "q": query[:24],
            }
            return rec, resp.headers.get("x-trace-id", "")
    except httpx.TimeoutException:
        return {"lat": (time.time() - t0) * 1000, "ok": False, "no_token": False,
                "handoff": False, "timeout": True, "status": "timeout", "q": query[:24]}, ""
    except Exception as e:
        return {"lat": (time.time() - t0) * 1000, "ok": False, "no_token": False,
                "handoff": False, "timeout": False, "status": f"err:{str(e)[:40]}", "q": query[:24]}, ""


async def virtual_user(idx: int, rounds: int, sem: asyncio.Semaphore,
                       results: list, feedbacks: list, timeout: int, downvote: float,
                       unique: bool = False):
    user_id = USERS[idx % len(USERS)]
    session_id = f"synthetic-{idx:03d}-{int(time.time())}"
    code = PROFILES[user_id]["verify_code"]
    async with httpx.AsyncClient() as client:
        for _ in range(rounds):
            name, convo = _pick()
            async with sem:
                for turn_i, q in enumerate(convo):
                    # 申购场景最后一轮补验证码 (PROFILES 里的固定码)
                    if name == "subscribe" and turn_i == len(convo) - 1:
                        q = f"{q} 验证码 {code}"
                    if unique:
                        # 打破语义缓存: 问题池只有 19 条, 反复跑会 100% 命中缓存,
                        # 于是 P50 变 2ms、sources 恒空 —— 测到的根本不是检索链路。
                        # 代价: 后缀会轻微扰动 BM25 词面命中, 指标比纯真实流量略噪。
                        q = f"{q}（编号{random.randint(1000, 9999)}）"
                    rec, trace_id = await _turn(client, session_id, user_id, q, timeout)
                    rec["scenario"] = name
                    results.append(rec)
                    if trace_id and random.random() < downvote:
                        feedbacks.append((session_id, user_id, trace_id, q))


async def agent_worker(client, token, results: list, done: asyncio.Event, stop_after: int):
    """模拟坐席: 轮询工单队列 → 接单 → AI 草稿 → 回复 → 关单 (让接单/结单时长可算)

    /queue 返回 open + taken, 只挑 open 的接 —— 否则重复 take 会吃 400。
    """
    h = {"Authorization": f"Bearer {token}"}
    me = {"agent_id": "synthetic"}
    closed = 0
    while not done.is_set() and closed < stop_after:
        try:
            r = await client.get(f"{API}/api/v1/handoff/queue", headers=h, timeout=20)
            for t in (r.json().get("tickets") or []):
                if t.get("status") != "open":
                    continue
                tid = t.get("id")
                if not tid:
                    continue
                tk = await client.post(f"{API}/api/v1/handoff/{tid}/take", headers=h, json=me, timeout=20)
                if tk.status_code != 200:
                    continue
                sug = await client.post(f"{API}/api/v1/handoff/{tid}/suggest", headers=h, timeout=60)
                draft = (sug.json() or {}).get("draft") or "已收到，我们尽快为您处理。"
                await client.post(f"{API}/api/v1/handoff/{tid}/reply", headers=h,
                                  json={"content": draft, "agent_id": "synthetic"}, timeout=20)
                await client.post(f"{API}/api/v1/handoff/{tid}/close", headers=h, json=me, timeout=20)
                closed += 1
                results.append({"ticket": tid, "ai_draft_len": len(draft)})
                if closed >= stop_after:
                    break
        except Exception:
            pass
        await asyncio.sleep(2)


# ────────────────────────────── 指标报告 ──────────────────────────────

def _pct(vals: list, p: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    return int(s[min(len(s) - 1, int(len(s) * p))])


def report(results: list, feedbacks: list, wall_s: float, agent_stats: list,
           srv: dict = None, since: float = None):
    from src.db.manager import db_manager

    n = len(results)
    ok = [r for r in results if r["ok"]]
    no_tok = [r for r in ok if r["no_token"]]
    lats = [r["lat"] for r in results]
    handoffs = [r for r in results if r.get("handoff")]
    timeouts = [r for r in results if r.get("timeout")]

    print("\n" + "=" * 62)
    print("📊 合成用户报告")
    print("=" * 62)
    print(f"  轮次 {n} | 墙钟 {wall_s:.1f}s | 反馈 {len(feedbacks)} 条")
    print("\n  ── 性能 ──")
    print(f"  QPS          {n / max(wall_s, 0.001):.2f}   (墙钟口径, 非 max(延迟))")
    print(f"  成功率       {len(ok)/max(n,1):.1%}    错误率 {1-len(ok)/max(n,1):.1%}"
          f"    超时率 {len(timeouts)/max(n,1):.1%}")
    print(f"  无 token 流   {len(no_tok)/max(len(ok),1):.1%}   (静态节点/缓存, 非缓存命中率)")
    if srv and srv.get("cache_hit_rate") is not None:
        print(f"  缓存命中率   {srv.get('cache_hit_rate')}%   (服务端口径, 以此为准)")
    if lats:
        print(f"  P50 {_pct(lats,0.50)}ms | P90 {_pct(lats,0.90)}ms | "
              f"P95 {_pct(lats,0.95)}ms | P99 {_pct(lats,0.99)}ms | 均值 {statistics.mean(lats):.0f}ms")
    else:
        print("  延迟       无数据")
    print(f"  触发转人工   {len(handoffs)} 轮 ({len(handoffs)/max(n,1):.1%})")

    cur = db_manager.conn.cursor()

    # ── 人机协同: 工单状态分布 / 接单与结单时长 ──
    # 只统计本次运行窗口内创建的工单 —— 否则会混进上一次跑剩下的 open/taken 工单,
    # 让「接单时长」被算成上次创建到这次接单的间隔 (实测冒出过 14.6 小时的假值)。
    cur.execute("SELECT status, COUNT(*) c FROM handoff_tickets GROUP BY status")
    dist = {r["status"]: r["c"] for r in cur.fetchall()}
    total_tickets = sum(dist.values())
    cur.execute("""SELECT created_at, taken_at, resolved_at FROM handoff_tickets
                   WHERE taken_at IS NOT NULL AND resolved_at IS NOT NULL""")
    rows = [r for r in cur.fetchall()
            if since is None or _parse_ts(r["created_at"]) >= since]
    take_s = [x for x in (_parse_ts(r["taken_at"]) - _parse_ts(r["created_at"]) for r in rows) if x >= 0]
    close_s = [x for x in (_parse_ts(r["resolved_at"]) - _parse_ts(r["created_at"]) for r in rows) if x >= 0]

    from src.config import settings
    sla_limit = settings.handoff_sla

    print("\n  ── 人机协同 ──")
    print(f"  工单总数 {total_tickets} | 状态分布 {dist or '{}'}")
    if srv and srv.get("handoff_rate") is not None:
        print(f"  转人工量 {srv.get('handoff_count')} | 转人工率 {srv.get('handoff_rate')}%   (服务端口径)")
    if take_s:
        sla = [x for x in take_s if x <= sla_limit]
        print(f"  平均接单时长 {statistics.mean(take_s):.1f}s | "
              f"SLA 达标率(≤{sla_limit}s) {len(sla)/len(take_s):.1%}")
    else:
        print(f"  平均接单时长 待定 (无已接单工单)")
    print(f"  平均结单时长 {statistics.mean(close_s):.1f}s" if close_s else "  平均结单时长 待定 (无已结单工单)")
    if agent_stats:
        print(f"  AI 草稿      生成 {len(agent_stats)} 条, "
              f"平均 {int(statistics.mean([a['ai_draft_len'] for a in agent_stats]))} 字")

    # ── badcase 闭环 ──
    cands = db_manager.query_badcases(limit=100000)
    reasons: dict = {}
    for row in cands:
        for r in db_manager._badcase_reasons(row):
            reasons[r] = reasons.get(r, 0) + 1
    cur.execute("SELECT COUNT(*) c FROM turn_traces")
    traces = cur.fetchone()["c"] or 0
    cur.execute("""SELECT triage_status s, COUNT(*) c FROM turn_traces
                   WHERE triage_status IS NOT NULL AND triage_status != ''
                   GROUP BY triage_status""")
    triage = {r["s"]: r["c"] for r in cur.fetchall()}

    print("\n  ── badcase 闭环 ──")
    print(f"  trace 总数 {traces} | 候选池 {len(cands)} | badcase 密度 {len(cands)/max(traces,1)*1000:.1f}‰")
    print(f"  原因分布     {reasons or '{}'}")
    print(f"  分诊分布     {triage or '{}'}")

    fails = [r for r in results if not r["ok"]][:3]
    for f in fails:
        print(f"  ⚠️  {f['status']} | {f['scenario']} | {f['q']}")
    print("=" * 62)
    return cands


def _parse_ts(s):
    """解析 time.ctime() 写入的 'Thu Sep 18 23:53:00 2026'。解析不了返回 0 并由调用方过滤。"""
    if not s:
        return 0.0
    try:
        from datetime import datetime
        return datetime.strptime(s, "%a %b %d %H:%M:%S %Y").timestamp()
    except (ValueError, TypeError):
        return -1e9          # 让差值变负, 被 x >= 0 过滤掉


# ────────────────────────────── 入口 ──────────────────────────────

async def login(client, username, password):
    r = await client.post(f"{API}/api/v1/auth/login",
                          json={"username": username, "password": password}, timeout=20)
    r.raise_for_status()
    return r.json()["token"]


async def amain(args):
    from src.config import settings

    results: list = []
    feedbacks: list = []
    agent_stats: list = []
    sem = asyncio.Semaphore(args.concurrency)
    done = asyncio.Event()

    async with httpx.AsyncClient() as client:
        token = None
        try:
            token = await login(client, settings.admin_username, settings.admin_password)
            print(f"✅ 已登录 ({settings.admin_username})")
        except Exception as e:
            print(f"⚠️  登录失败, 跳过坐席模拟与监控接口: {e}")

        agent_task = None
        if token and not args.no_agent:
            agent_task = asyncio.create_task(
                agent_worker(client, token, agent_stats, done, args.users * 2))

        t_start = time.time()          # ← QPS 用墙钟, 不是 max(单请求延迟)
        await asyncio.gather(*[
            virtual_user(i, args.rounds, sem, results, feedbacks, args.timeout,
                         args.downvote, args.unique)
            for i in range(args.users)
        ])

        # 点踩 (制造 downvote 型 badcase), 串行发避免打爆
        for session_id, user_id, trace_id, q in feedbacks:
            try:
                await client.post(f"{API}/api/v1/chat/feedback",
                                  json={"session_id": session_id, "rating": -1, "user_id": user_id,
                                        "trace_id": trace_id, "comment": f"synthetic:{q[:30]}"},
                                  timeout=20)
            except Exception:
                pass

        if agent_task:
            await asyncio.sleep(3)      # 让坐席把队列里的工单收完
            done.set()
            await agent_task

        wall_s = time.time() - t_start

        # 服务端侧指标 (SLI + 转人工量率) —— 与脚本自算的互相印证
        srv = {}
        if token:
            h = {"Authorization": f"Bearer {token}"}
            try:
                sli = (await client.get(f"{API}/api/v1/monitor/sli?window=1h", headers=h,
                                        timeout=20)).json()
                srv.update(sli)          # cache_hit_rate / p50 / p90 / p99 在 SLI 里
                print(f"\n  服务端 SLI(1h): QPS {sli.get('qps')} | 成功率 {sli.get('success_rate')}"
                      f" | 缓存命中 {sli.get('cache_hit_rate')} | P50 {sli.get('p50_ms')}ms"
                      f" | P90 {sli.get('p90_ms')}ms | P99 {sli.get('p99_ms')}ms")
            except Exception as e:
                print(f"  ⚠️  SLI 读取失败: {e}")
            try:
                srv.update((await client.get(f"{API}/api/v1/monitor/metrics",
                                             headers=h, timeout=20)).json())   # handoff_count/rate
            except Exception as e:
                print(f"  ⚠️  metrics 读取失败: {e}")

    cands = report(results, feedbacks, wall_s, agent_stats, srv, since=t_start)

    if args.promote and cands:
        await promote(cands, args.promote)


async def promote(cands: list, n: int):
    """提升 N 条 badcase 为 Golden。会写 data/eval/finance_qa_golden.jsonl —— 需 --promote 显式开启。"""
    from src.config import settings
    async with httpx.AsyncClient() as client:
        token = await login(client, settings.admin_username, settings.admin_password)
        h = {"Authorization": f"Bearer {token}"}
        ok = 0
        for row in cands[:n]:
            tid = row.get("trace_id")
            try:
                r = await client.post(f"{API}/api/v1/badcases/{tid}/promote", headers=h, timeout=20)
                if r.status_code == 200:
                    ok += 1
                    print(f"  ↑ 已提升 {tid} → {r.json().get('id', r.json())}")
            except Exception as e:
                print(f"  ⚠️  提升 {tid} 失败: {e}")
        print(f"\n  Golden 转正 {ok} 条 —— 记得人工补 evidence 后再跑回归。")


def main():
    p = argparse.ArgumentParser(description="合成用户压测 + badcase 冷启动")
    p.add_argument("--users", type=int, default=12, help="虚拟用户数")
    p.add_argument("--rounds", type=int, default=3, help="每个用户跑几轮会话")
    p.add_argument("--concurrency", type=int, default=6, help="并发上限 (对齐 processing_slot=6)")
    p.add_argument("--timeout", type=int, default=240, help="单轮超时秒")
    p.add_argument("--downvote", type=float, default=0.15, help="点踩比例")
    p.add_argument("--no-agent", action="store_true", help="不模拟坐席接单")
    p.add_argument("--unique", action="store_true",
                   help="给每条 query 加随机后缀打破语义缓存 (测真实检索链路必开)")
    p.add_argument("--promote", type=int, default=0, help="提升 N 条 badcase 为 Golden (写文件)")
    p.add_argument("--report-only", action="store_true", help="只读库出报告")
    args = p.parse_args()
    random.seed(args.users)

    if args.report_only:
        report([], [], 0.001, [])
        return
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
