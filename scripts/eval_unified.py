"""
统一评测报告：功能指标 + 性能指标 一页展示

整合:
  - eval_finance.py: 离线（证据覆盖/chunk信息命中/路径一致/图谱触发）+ 在线（意图准确/合规通过）
  - eval_retrieval.py: Recall@K + MRR
  - benchmark_chat_latency.py: p50/p90/p95 延迟 + TTFB

用法:
  .venv/Scripts/python.exe scripts/eval_unified.py
  .venv/Scripts/python.exe scripts/eval_unified.py --with-answers --limit 20
  .venv/Scripts/python.exe scripts/eval_unified.py --skip-performance
输出: data/eval/report_unified.json + 控制台打印
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUT = os.path.join(ROOT, "data", "eval", "report_unified.json")


def run_retrieval_eval() -> dict:
    """跑检索评测：Recall@K + MRR"""
    import eval_retrieval
    # eval_retrieval 的 main 会打印并返回结果，这里直接调用核心函数
    # 简化：跑默认配置
    try:
        result = eval_retrieval.run_eval(limit=0, top_k=5, rerank="auto")
        return result
    except Exception as e:
        print(f"检索评测跳过: {e}")
        return {"error": str(e)}


def run_finance_eval(with_answers: bool = False, limit: int = 0) -> dict:
    """跑金融Golden QA评测：离线 + 在线"""
    import eval_finance
    rows = eval_finance.load_golden()
    if limit:
        rows = rows[:limit]

    report = {"golden_n": len(rows)}
    report["offline"] = eval_finance.offline_metrics(rows)
    if with_answers:
        report["online"] = eval_finance.answer_metrics(rows, limit or len(rows))
    return report


def run_performance_benchmark() -> dict:
    """跑性能基准：p50/p90/p95 + TTFB"""
    try:
        import benchmark_chat_latency
        result = benchmark_chat_latency.run_benchmark(rounds=5)
        return result
    except Exception as e:
        print(f"性能基准跳过: {e}")
        return {"error": str(e)}


def print_report(report: dict):
    """打印统一评测报告"""
    print("\n" + "=" * 70)
    print("  统一评测报告")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 功能指标
    finance = report.get("finance", {})
    offline = finance.get("offline", {})
    online = finance.get("online", {})

    print("\n【功能指标】")
    print(f"  Golden QA 数量: {offline.get('n', 'N/A')}")
    print(f"  证据覆盖率:     {offline.get('evidence_coverage', 'N/A')}")
    print(f"  chunk信息命中率: {offline.get('chunk_info_hit_rate', 'N/A')}")
    print(f"  路径结构一致率: {offline.get('path_structure_consistency', 'N/A')}")
    print(f"  图谱触发率:     {offline.get('graph_trigger_ratio', 'N/A')}")

    if online:
        print(f"\n  意图准确率:     {online.get('intent_accuracy', 'N/A')}")
        print(f"  合规通过率:     {online.get('compliance_pass_rate', 'N/A')}")

    # 检索指标
    retrieval = report.get("retrieval", {})
    if retrieval and "error" not in retrieval:
        print("\n【检索指标】")
        for k, v in retrieval.items():
            if k != "rows" and isinstance(v, (int, float)):
                print(f"  {k}: {v}")

    # 性能指标
    perf = report.get("performance", {})
    if perf and "error" not in perf:
        print("\n【性能指标】")
        for k, v in perf.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v}")
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        print(f"  {k}.{k2}: {v2}")

    # 综合评分
    print("\n【综合评估】")
    scores = []
    if offline.get("evidence_coverage"):
        scores.append(("证据覆盖", offline["evidence_coverage"]))
    if offline.get("chunk_info_hit_rate"):
        scores.append(("chunk信息命中", offline["chunk_info_hit_rate"]))
    if online.get("intent_accuracy"):
        scores.append(("意图准确", online["intent_accuracy"]))
    if online.get("compliance_pass_rate"):
        scores.append(("合规通过", online["compliance_pass_rate"]))

    if scores:
        avg = sum(s[1] for s in scores) / len(scores)
        print(f"  功能综合得分: {avg:.3f} ({len(scores)} 项指标平均)")
        for name, score in scores:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    {name:12s} {bar} {score:.3f}")

    print("\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-answers", action="store_true", help="跑在线评测（需LLM）")
    ap.add_argument("--limit", type=int, default=0, help="在线评测条数限制")
    ap.add_argument("--skip-retrieval", action="store_true", help="跳过检索评测")
    ap.add_argument("--skip-performance", action="store_true", help="跳过性能基准")
    args = ap.parse_args()

    start = time.time()
    report = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    print("跑金融 Golden QA 评测...")
    report["finance"] = run_finance_eval(with_answers=args.with_answers, limit=args.limit)

    if not args.skip_retrieval:
        print("跑检索评测...")
        report["retrieval"] = run_retrieval_eval()

    if not args.skip_performance:
        print("跑性能基准...")
        report["performance"] = run_performance_benchmark()

    report["duration_sec"] = round(time.time() - start, 1)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"\n报告已保存: {OUTPUT}")
    print(f"总耗时: {report['duration_sec']}s")


if __name__ == "__main__":
    main()
