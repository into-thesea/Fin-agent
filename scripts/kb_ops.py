"""
知识库运营工具 — 低置信度审核队列 + 热点问题分析

功能:
  1. low_confidence_queue: 从评测报告中筛选低置信度/证据未覆盖的用例, 生成审核队列
  2. hotspot_analysis: 统计 Golden QA 中各意图分布和未命中问题, 识别知识库补充优先级
  3. kb_gap_report: 对比知识库文档和 Golden QA, 找出覆盖不足的主题

用法:
  python scripts/kb_ops.py --queue          # 生成低置信度审核队列
  python scripts/kb_ops.py --hotspot        # 热点问题分析
  python scripts/kb_ops.py --gap            # 知识库覆盖缺口报告
  python scripts/kb_ops.py --all            # 全部运行
"""

import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK, 强制 UTF-8 输出避免 emoji 打印报错
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "finance_qa_golden.jsonl"
EVAL_REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "report_finance.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kb_ops"


def load_golden() -> list:
    """加载 Golden QA"""
    if not GOLDEN_PATH.exists():
        return []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_report() -> dict:
    """加载评测报告"""
    if not EVAL_REPORT_PATH.exists():
        return {}
    with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def low_confidence_queue(threshold: float = 0.7) -> list:
    """
    从评测报告中筛选低置信度/证据未覆盖的用例

    筛选条件 (满足任一):
      - confidence < threshold
      - evidence_covered == False
      - chunk_info_hit == False (文档命中但信息没命中)
    """
    report = load_eval_report()
    cases = report.get("cases", []) or report.get("details", [])

    queue = []
    for case in cases:
        conf = case.get("confidence", 1.0)
        evidence = case.get("evidence_covered", True)
        chunk_hit = case.get("chunk_info_hit", True)
        intent = case.get("intent", "unknown")
        question = case.get("question", case.get("query", ""))

        if conf < threshold or not evidence or not chunk_hit:
            queue.append({
                "id": case.get("id", ""),
                "question": question,
                "intent": intent,
                "confidence": conf,
                "evidence_covered": evidence,
                "chunk_info_hit": chunk_hit,
                "reason": _classify_reason(conf, evidence, chunk_hit),
                "suggested_action": _suggest_action(intent, evidence, chunk_hit),
            })

    # 按严重程度排序: 证据未覆盖 > chunk未命中 > 低置信度
    severity = {"evidence_missing": 0, "chunk_missing": 1, "low_confidence": 2}
    queue.sort(key=lambda x: severity.get(x["reason"], 3))
    return queue


def _classify_reason(conf: float, evidence: bool, chunk_hit: bool) -> str:
    if not evidence:
        return "evidence_missing"
    if not chunk_hit:
        return "chunk_missing"
    if conf < 0.7:
        return "low_confidence"
    return "other"


def _suggest_action(intent: str, evidence: bool, chunk_hit: bool) -> str:
    if not evidence:
        return "补充知识库文档: 该问题无相关知识源"
    if not chunk_hit:
        return "优化分块/检索: 文档命中但关键信息未召回"
    return "优化 Prompt/few-shot: 意图分类或生成置信度低"


def hotspot_analysis(top_n: int = 20) -> dict:
    """
    统计 Golden QA 中各意图分布, 识别高频问题主题

    Returns:
        {
            "intent_distribution": {intent: count},
            "top_questions": [(question, count), ...],
            "coverage_gaps": [intent with < 3 samples],
        }
    """
    golden = load_golden()
    if not golden:
        return {"intent_distribution": {}, "top_questions": [], "coverage_gaps": []}

    # 意图分布
    intent_counts = Counter(g.get("intent", "unknown") for g in golden)

    # 产品实体分布 (识别哪些产品被问得多)
    product_keywords = [
        "稳盈添利", "平衡增利", "私银聚享", "大额存单", "结构性存款",
        "日日盈", "安鑫纯债", "远见成长", "增额终身寿", "存款保险",
    ]
    product_counts = Counter()
    for g in golden:
        q = g.get("question", "")
        for kw in product_keywords:
            if kw in q:
                product_counts[kw] += 1

    # 覆盖不足的意图 (< 3 条样本)
    coverage_gaps = [intent for intent, count in intent_counts.items() if count < 3]

    return {
        "total": len(golden),
        "intent_distribution": dict(intent_counts.most_common()),
        "product_distribution": dict(product_counts.most_common()),
        "coverage_gaps": coverage_gaps,
        "recommendation": _hotspot_recommendation(intent_counts, product_counts),
    }


def _hotspot_recommendation(intent_counts: Counter, product_counts: Counter) -> list:
    """根据分布给出知识库补充建议"""
    recs = []
    # 高频意图但样本少 → 补充 Golden QA
    for intent, count in intent_counts.most_common(3):
        if count < 5:
            recs.append(f"高频意图 [{intent}] 仅 {count} 条样本, 建议补充到 10+ 条")
    # 高频产品 → 确保知识库覆盖
    for product, count in product_counts.most_common(3):
        recs.append(f"高频产品 [{product}] 被问 {count} 次, 确认知识库文档完整")
    return recs


def kb_gap_report() -> dict:
    """
    对比知识库文档和 Golden QA, 找出覆盖不足的主题

    检查:
      1. Golden QA 中提到的产品是否都有对应知识库文档
      2. 各意图类型是否有对应的知识源
    """
    golden = load_golden()
    kb_dir = PROJECT_ROOT / "data" / "finance_kb"

    # 列出知识库文档
    kb_docs = []
    if kb_dir.exists():
        kb_docs = [f.name for f in kb_dir.iterdir() if f.suffix in (".md", ".txt", ".jsonl")]

    # Golden QA 中提到的产品
    products_in_qa = set()
    product_keywords = [
        "稳盈添利30天", "稳盈添利90天", "平衡增利180天", "私银聚享",
        "大额存单", "结构性存款", "日日盈", "安鑫纯债", "远见成长", "增额终身寿",
    ]
    for g in golden:
        q = g.get("question", "")
        for kw in product_keywords:
            if kw in q:
                products_in_qa.add(kw)

    # 检查每个产品是否有对应文档
    product_docs = {}
    for product in products_in_qa:
        matched = [d for d in kb_docs if product[:4] in d]
        product_docs[product] = matched if matched else ["⚠️ 未找到对应文档"]

    return {
        "kb_doc_count": len(kb_docs),
        "kb_docs": kb_docs,
        "products_in_qa": list(products_in_qa),
        "product_docs": product_docs,
        "missing_products": [p for p, docs in product_docs.items() if "⚠️" in docs[0]],
    }


def main():
    parser = argparse.ArgumentParser(description="知识库运营工具")
    parser.add_argument("--queue", action="store_true", help="生成低置信度审核队列")
    parser.add_argument("--hotspot", action="store_true", help="热点问题分析")
    parser.add_argument("--gap", action="store_true", help="知识库覆盖缺口报告")
    parser.add_argument("--all", action="store_true", help="全部运行")
    parser.add_argument("--threshold", type=float, default=0.7, help="低置信度阈值")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.queue or args.all:
        queue = low_confidence_queue(args.threshold)
        out = OUTPUT_DIR / "low_confidence_queue.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)
        print(f"[低置信度队列] {len(queue)} 条 → {out}")
        for item in queue[:5]:
            print(f"  - [{item['reason']}] {item['question'][:50]}...")

    if args.hotspot or args.all:
        hotspot = hotspot_analysis()
        out = OUTPUT_DIR / "hotspot_analysis.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(hotspot, f, ensure_ascii=False, indent=2)
        print(f"\n[热点分析] 共 {hotspot['total']} 条 Golden QA → {out}")
        print("  意图分布:")
        for intent, count in hotspot["intent_distribution"].items():
            print(f"    {intent}: {count}")
        for rec in hotspot["recommendation"]:
            print(f"  建议: {rec}")

    if args.gap or args.all:
        gap = kb_gap_report()
        out = OUTPUT_DIR / "kb_gap_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(gap, f, ensure_ascii=False, indent=2)
        print(f"\n[覆盖缺口] 知识库 {gap['kb_doc_count']} 篇文档 → {out}")
        if gap["missing_products"]:
            print(f"  ⚠️ 缺少文档的产品: {', '.join(gap['missing_products'])}")
        else:
            print("  ✅ 所有产品均有对应文档")


if __name__ == "__main__":
    main()
