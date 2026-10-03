"""闸门阈值标定: 在 golden 上扫描 T/S, 产出覆盖率-错误率表.

用法:
    python scripts/calibrate_fast_path.py
    python scripts/calibrate_fast_path.py --limit 20

**阈值不跨模型迁移** —— 换 router 模型后必须重跑本脚本 (见 spec §4.2)。
扫描期间 S 固定、T 单独扫, 不同时调两个参数, 否则等于对着评测集拟合。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.core.fast_path as fp
from src.agents.router_agent import RouterAgent
from src.tools.registry import retrieve_knowledge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "eval", "fast_path_calibration.json")


def sweep(rows: list, Ts: list, Ss: list) -> list:
    """扫 T/S 组合 → [{T, S, fast, coverage, wrong, wrong_rate}]"""
    out = []
    saved_T, saved_S = fp.CONFIDENCE_THRESHOLD, fp.KB_SCORE_THRESHOLD
    try:
        for T in Ts:
            for S in Ss:
                fp.CONFIDENCE_THRESHOLD = T
                fp.KB_SCORE_THRESHOLD = S
                passed = []
                for r in rows:
                    ctx = {"local": [{"score": r["top1"]}]}
                    ok, _ = fp.evaluate(r["intent"], r["confidence"], ctx,
                                        r["query"], r["strong"])
                    if ok:
                        passed.append(r)
                wrong = sum(1 for r in passed if not r["correct"])
                out.append({
                    "T": T, "S": S, "fast": len(passed),
                    "coverage": round(len(passed) / max(1, len(rows)), 4),
                    "wrong": wrong,
                    "wrong_rate": round(wrong / max(1, len(passed)), 4),
                })
    finally:
        fp.CONFIDENCE_THRESHOLD, fp.KB_SCORE_THRESHOLD = saved_T, saved_S
    return out


def collect(path: str) -> list:
    """逐条跑 router + 检索, 采集标定所需的原始字段"""
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    router = RouterAgent()
    rows = []
    for i, rec in enumerate(recs):
        q = rec["question"]
        res = router.route(q)
        _strong_result, strong = router.strong_signal_rule(q)
        kb = retrieve_knowledge(q, 8)
        rows.append({
            "id": rec["id"], "query": q, "expected": rec["intent"],
            "intent": res.intent.value, "confidence": float(res.confidence),
            "top1": fp.top1_score(kb["contexts"]), "strong": strong,
            "personal": any(m in q for m in fp.PRIVATE_DATA_MARKERS),
            "correct": res.intent.value == rec["intent"],
        })
        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{len(recs)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(ROOT, "data", "eval", "finance_qa_golden.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = collect(args.eval)
    if args.limit:
        rows = rows[:args.limit]
    table = sweep(rows, Ts=[0.90, 0.95, 0.98, None], Ss=[0.55, 0.60, 0.65])
    print(f"\n样本 {len(rows)} 条")
    print("   T      S     快答数  覆盖   意图错  错误率")
    for r in table:
        t = "off " if r["T"] is None else f'{r["T"]:.2f}'
        print(f'  {t}  {r["S"]:.2f}   {r["fast"]:3d}   {r["coverage"]:5.0%}    '
              f'{r["wrong"]:2d}   {r["wrong_rate"]:5.1%}')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"rows": rows, "sweep": table}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n原始数据已写入 {OUT}")


if __name__ == "__main__":
    main()
