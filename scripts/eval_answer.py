"""
P1: 答案质量评测脚本 — LLM-as-Judge

对 data/eval/qa_golden.jsonl 中每条问题：
  1. 用系统实际管道 (AnalystAgent.analyze) 生成答案
  2. 用 LLM (create_client 主模型) 当 judge，对照参考答案打分
     accuracy(正确性) / faithfulness(忠实度) / refuse_correct(拒答正确) / hallucinated(幻觉)

指标:
  平均正确率 / 平均忠实度 / 幻觉率 / 拒答正确率 / 通过率(accuracy>=60)

用法:
  python scripts/eval_answer.py                    # 全量评测
  python scripts/eval_answer.py --limit 10         # 只跑前 10 条
  python scripts/eval_answer.py --offset 10 --limit 10   # 分段续跑
  python scripts/eval_answer.py --skip-judge       # 只跑答案不判分 (排查用)
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK, 强制 UTF-8 输出避免 emoji 打印报错
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.llm.llm_client import create_client

DEFAULT_EVAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval", "qa_golden.jsonl"
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(DEFAULT_EVAL), "report.json"
)

JUDGE_PROMPT = """你是一名严格的 AI 客服回答质量评审。请评审系统回答是否合格。

【用户问题】{question}
【权威参考答案】{reference}
【系统回答】{answer}

请按以下维度打分（分数 0-100 整数）：
1. accuracy 正确性：回答是否正确回应了问题，事实是否与参考答案一致
2. faithfulness 忠实度：回答中的论断是否都有参考答案支撑（无支撑的论断算不忠实）
3. refuse_correct 拒答正确性：对知识库外/无关问题，系统是否诚实说明无法回答或引导转人工（而不是编造）？是则 true，否则 false
4. hallucinated 幻觉：回答是否捏造了参考答案中不存在的关键事实（如具体数字/规则）？是则 true，否则 false

只输出一个 JSON 对象，不要输出其他文字：
{{"accuracy": 0, "faithfulness": 0, "refuse_correct": true, "hallucinated": false, "reason": "一句话说明"}}
"""


def load_records(path: str) -> list:
    if not os.path.exists(path):
        print(f"❌ 评测集不存在: {path}")
        sys.exit(1)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  ⚠️ 跳过坏行: {line[:60]}")
    return records


def run_judge(client, question: str, reference: str, answer: str) -> dict | None:
    """LLM-as-Judge，失败时返回 None"""
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer[:2000])
    for attempt in range(2):
        try:
            raw = client.chat(
                messages=[
                    {"role": "system", "content": "你是一个严格的评测助手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            # 提取 JSON 子串（容忍多余文字）
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"judge 未输出 JSON: {raw[:120]}")
            data = json.loads(raw[start:end + 1])
            return {
                "accuracy": int(data.get("accuracy", 0)),
                "faithfulness": int(data.get("faithfulness", 0)),
                "refuse_correct": bool(data.get("refuse_correct", False)),
                "hallucinated": bool(data.get("hallucinated", True)),
                "reason": str(data.get("reason", "")),
            }
        except Exception as e:
            if attempt == 0:
                continue
            print(f"  ⚠️ judge 失败: {e}")
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="答案质量评测 (LLM-as-Judge)")
    parser.add_argument("--eval", default=DEFAULT_EVAL, help="评测集路径")
    parser.add_argument("--out", default=DEFAULT_OUT, help="报告输出路径")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条")
    parser.add_argument("--skip-judge", action="store_true", help="只跑答案不判分")
    parser.add_argument("--sleep", type=float, default=0, help="每条之间间隔秒数 (防限流)")
    parser.add_argument("--backend", choices=["pipeline", "graph"], default="pipeline",
                        help="系统后端: pipeline=AnalystAgent(默认) / graph=LangGraph 状态机(P3)")
    args = parser.parse_args()

    records = load_records(args.eval)
    if args.offset:
        records = records[args.offset:]
    if args.limit:
        records = records[:args.limit]
    if not records:
        print("❌ 评测集为空")
        sys.exit(1)
    print(f"📋 评测集: {len(records)} 条问题 (共 {args.offset + len(records)}/{args.offset + args.limit or ''})")

    if args.backend == "graph":
        from src.graph.cs_graph import run
        agent = None
    else:
        from src.analyst_agent import AnalystAgent
        agent = AnalystAgent()
    client = None if args.skip_judge else create_client()
    if agent:
        print(f"🔧 后端: pipeline (AnalystAgent)")
    else:
        print(f"🔧 后端: graph (LangGraph 状态机)")

    per_record = []
    for i, rec in enumerate(records, 1):
        q = rec["question"]
        t0 = time.time()
        try:
            if args.backend == "graph":
                result = run(q, session_id=f"eval_{rec['id']}")
            else:
                result = agent.analyze(q)
            answer = result.get("answer", "")
            handoff = bool(result.get("handoff"))
        except Exception as e:
            print(f"  ❌ 系统出错 [{q[:30]}]: {str(e)[:120]}")
            per_record.append({"id": rec["id"], "question": q, "error": str(e)[:200]})
            continue
        dt = time.time() - t0
        print(f"  [{i}/{len(records)}] {rec['category']:8s} {q[:24]:26s} {dt:5.1f}s handoff={handoff}")

        scores = None
        if not args.skip_judge:
            scores = run_judge(client, q, rec.get("reference", ""), answer)
            if scores:
                print(f"       acc={scores['accuracy']:3d} faith={scores['faithfulness']:3d} "
                      f"refuse={scores['refuse_correct']} hal={scores['hallucinated']} | {scores['reason'][:50]}")

        per_record.append({
            "id": rec["id"],
            "category": rec["category"],
            "question": q,
            "answer": answer,
            "handoff": handoff,
            "scores": scores,
        })
        if args.sleep:
            time.sleep(args.sleep)
    if agent:
        agent.close()

    # ── 聚合 ──────────────────────────────
    scored = [r for r in per_record if r.get("scores")]
    n = len(scored)
    if n == 0:
        print("\n⚠️ 无有效评分 (是否用了 --skip-judge 或全部失败)")
        if args.out:
            save_report(args.out, {"total": len(per_record), "per_record": per_record})
        sys.exit(0)

    overall = {
        "accuracy_avg": sum(r["scores"]["accuracy"] for r in scored) / n,
        "faithfulness_avg": sum(r["scores"]["faithfulness"] for r in scored) / n,
        "hallucination_rate": sum(r["scores"]["hallucinated"] for r in scored) / n,
        "refuse_correct_rate": sum(r["scores"]["refuse_correct"] for r in scored) / n,
        "pass_rate": sum(r["scores"]["accuracy"] >= 60 for r in scored) / n,
        "total": n,
    }

    print("\n" + "=" * 52)
    print("📊 答案质量评测报告")
    print("=" * 52)
    print(f"  平均正确率     {overall['accuracy_avg']:6.1f}")
    print(f"  平均忠实度     {overall['faithfulness_avg']:6.1f}")
    print(f"  幻觉率         {overall['hallucination_rate']:6.1%}")
    print(f"  拒答正确率     {overall['refuse_correct_rate']:6.1%}")
    print(f"  通过率(≥60)    {overall['pass_rate']:6.1%}")
    print("  " + "-" * 48)
    print("  按类别:")
    cats = {}
    for r in scored:
        cats.setdefault(r["category"], []).append(r["scores"]["accuracy"])
    for cat, scores in cats.items():
        print(f"    {cat:8s}  n={len(scores):2d}  acc={sum(scores)/len(scores):6.1f}")
    print("=" * 52)

    if args.out:
        report = {"overall": overall, "per_record": per_record, "eval_file": args.eval}
        save_report(args.out, report)


def save_report(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 报告已保存: {path}")


if __name__ == "__main__":
    main()
