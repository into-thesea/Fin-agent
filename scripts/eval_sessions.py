"""多轮会话评测 —— 走 chat.py 的会话循环, 覆盖单轮评测测不到的部分

为什么需要这个脚本:
  cs_graph.run() 是无状态捷径, 不读会话槽位、不写回状态;
  真实用户走的是 chat.py 的循环 —— 取状态 → 指代改写 → 跑图(带会话槽位) → 写回状态。
  eval_finance.py --with-answers 用的是 run(), 因此测不到:
    指代消解 / 槽位跨轮继承 / 情绪累积 / 状态写回。

本脚本复刻 chat.py 的循环 (不含缓存与 SSE), 按 data/eval/finance_sessions.jsonl
里的会话逐轮连续提问, 记录每轮意图、改写结果、情绪与合规。

用法:
  .venv/Scripts/python.exe scripts/eval_sessions.py
  .venv/Scripts/python.exe scripts/eval_sessions.py --file data/eval/finance_sessions.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SESSIONS = os.path.join(PROJECT_ROOT, "data", "eval", "finance_sessions.jsonl")
REPORT = os.path.join(PROJECT_ROOT, "data", "eval", "report_sessions.json")

USER_ID = "u001"


def run_turn(session_id: str, query: str) -> dict:
    """复刻 chat.py 的单轮循环: 取状态 → 指代改写 → 跑图(带会话槽位) → 写回状态"""
    from src.core.dialog_state import get_state_manager
    from src.llm.query_rewriter import rewrite_query
    from src.graph.cs_graph import run

    sm = get_state_manager()
    state = sm.get_or_create(session_id)

    # 图运行前快照: 这些就是本轮真正喂给图的会话上下文
    before = {
        "emotion": state.emotion,
        "negative_turns": state.negative_turns,
        "turn_count": state.turn_count,
        "slots": dict(state.slots),
    }

    rewritten = rewrite_query(query, state)
    t0 = time.time()
    result = run(rewritten, session_id=session_id, user_id=USER_ID,
                 slots=dict(state.slots))
    latency = time.time() - t0

    sm.update(session_id, rewritten, result.get("answer", "") or "",
              entities=result.get("entities"), intent=result.get("intent"))

    return {"rewritten": rewritten, "result": result, "before": before,
            "after_emotion": state.emotion, "latency": latency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_SESSIONS)
    ap.add_argument("--out", default=REPORT)
    args = ap.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        sessions = [json.loads(l) for l in f if l.strip()]

    from src.core.compliance import finalize, check

    rows = []
    intent_ok = intent_n = 0
    comp_ok = comp_n = 0
    rewrite_ok = rewrite_n = 0
    followup_ok = followup_n = 0

    for sess in sessions:
        sid = sess["session_id"]
        print(f"\n[{sid}] {sess.get('note', '')}")
        for i, turn in enumerate(sess["turns"], 1):
            q = turn["query"]
            try:
                out = run_turn(sid, q)
            except Exception as e:
                print(f"  {i}. {q[:30]}  ⚠️ 异常: {str(e)[:80]}")
                rows.append({"session": sid, "turn": i, "query": q, "error": str(e)[:200]})
                continue

            res = out["result"]
            got = res.get("intent", "")
            exp = turn.get("expect_intent", "")
            hit = got == exp
            intent_n += 1
            intent_ok += hit

            ans = finalize(res.get("answer", "") or "")
            c = check(ans)
            comp_n += 1
            passed = not c["pii"]
            comp_ok += passed

            # 指代消解: 期望产品名出现在改写后的问题里
            prod = turn.get("expect_rewrite_product")
            rw_hit = None
            if prod:
                rewrite_n += 1
                rw_hit = prod in out["rewritten"]
                rewrite_ok += rw_hit

            is_followup = bool(turn.get("needs_context"))
            if is_followup:
                followup_n += 1
                followup_ok += hit

            mark = "OK " if hit else "MISS"
            print(f"  {i}. [{mark}] {q[:28]} → {got or '(空)'} (期望 {exp}) {out['latency']:.1f}s")
            if out["rewritten"] != q:
                print(f"       改写: {out['rewritten'][:46]}")
            if prod:
                print(f"       指代消解: {'命中' if rw_hit else '未消解'}")
            if res.get("handoff"):
                print("       → 转人工")

            rows.append({
                "session": sid, "turn": i, "query": q,
                "rewritten": out["rewritten"], "intent": got,
                "expect_intent": exp, "intent_hit": hit,
                "compliance_pass": passed, "handoff": bool(res.get("handoff")),
                "needs_context": is_followup,
                "rewrite_product_hit": rw_hit,
                "emotion_before": out["before"]["emotion"],
                "emotion_after": out["after_emotion"],
                "negative_turns_before": out["before"]["negative_turns"],
                "turn_count_before": out["before"]["turn_count"],
                "latency": round(out["latency"], 2),
            })

    report = {
        "sessions": len(sessions),
        "turns": len(rows),
        "intent_accuracy": round(intent_ok / intent_n, 3) if intent_n else None,
        "compliance_pass_rate": round(comp_ok / comp_n, 3) if comp_n else None,
        "followup_intent_accuracy": round(followup_ok / followup_n, 3) if followup_n else None,
        "followup_turns": followup_n,
        "coreference_resolved_rate": round(rewrite_ok / rewrite_n, 3) if rewrite_n else None,
        "coreference_cases": rewrite_n,
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 46)
    print(f"会话数 {report['sessions']} / 轮次 {report['turns']}")
    print(f"  多轮意图准确率      {report['intent_accuracy']}")
    print(f"  其中续轮(需上下文)  {report['followup_intent_accuracy']}  ({report['followup_turns']} 轮)")
    print(f"  合规通过率          {report['compliance_pass_rate']}")
    print(f"  指代消解成功率      {report['coreference_resolved_rate']}  ({report['coreference_cases']} 例)")
    print(f"报告: {args.out}")


if __name__ == "__main__":
    main()
