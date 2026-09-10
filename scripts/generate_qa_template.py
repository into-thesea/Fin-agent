"""
基于实体的模板化 QA 生成 — 免 LLM 最快路径

用 entities_extracted.jsonl 中每个 chunk 已标注的 (Company, Metric, Time) 实体,
套用金融问答模板生成评测问题, golden chunk = 实体所在 chunk (锚定零成本)。

适用场景: LLM API 不可用时 (如当前网络无法访问 Gemini), 先跑通评测闭环。
网络恢复 / 配置好 DeepSeek key 后, 可用 scripts/generate_qa.py (LLM 生成)
补充更自然、跨块的问题。

用法:
    python scripts/generate_qa_template.py                 # 默认 40 条
    python scripts/generate_qa_template.py --limit 60 --out scripts/eval_qa.jsonl
"""

import sys
import os
import re
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK, 强制 UTF-8 输出避免 emoji 打印报错
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.infra.paths import CHUNKS_PROCESSED_PATH, OUTPUT_ANALYSIS_DIR

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_qa.jsonl")
ENTITIES_EXTRACTED_PATH = os.path.join(OUTPUT_ANALYSIS_DIR, "entities_extracted.jsonl")

# 问题关键实体的类型 (用于回标校验: 这些词必须出现在 chunk 原文中)
KEY_TYPES = ("Company", "Metric", "FinancialMetric")


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


CN_RE = re.compile(r"[一-鿿]")


def prefer_cn(names):
    """优先含中文的实体名 (更贴近文档表述与真实用户提问, 提升检索命中)"""
    cn = [n for n in names if CN_RE.search(n)]
    return cn or names


def prefer_year(times):
    """优先纯年份 (2024), 避免过长日期形式 (2024-12-31)"""
    years = [t for t in times if re.fullmatch(r"20\d{2}", t)]
    return years or times


def gen_for_chunk(entities):
    """为一个 chunk 的实体生成 1 个最具体的问题 (Company+Metric+Time 优先)"""
    comps = [e["name"] for e in entities if e.get("type") == "Company"]
    metrics = [e["name"] for e in entities if e.get("type") in ("Metric", "FinancialMetric")]
    times = [e["name"] for e in entities if e.get("type") == "Time"]
    inds = [e["name"] for e in entities if e.get("type") == "Industry"]

    # 过滤过长/无意义的实体名, 优先中文名与纯年份
    comps = prefer_cn([c for c in comps if 1 < len(c) <= 30])
    metrics = prefer_cn([m for m in metrics if 1 < len(m) <= 30])
    times = prefer_year([t for t in times if len(t) <= 12])
    inds = prefer_cn([i for i in inds if len(i) <= 20])

    if comps and metrics and times:
        return f"{comps[0]}在{times[0]}的{metrics[0]}是多少？"
    if comps and metrics:
        return f"{comps[0]}的{metrics[0]}是多少？"
    if comps and times and inds:
        return f"{comps[0]}在{times[0]}的{inds[0]}业务情况如何？"
    if comps and inds:
        return f"{comps[0]}的{inds[0]}业务情况如何？"
    if comps:
        return f"{comps[0]}是一家什么样的公司？"
    if metrics and times:
        return f"{times[0]}的{metrics[0]}是多少？"
    return None


def main():
    parser = argparse.ArgumentParser(description="基于实体的模板化 QA 生成 (免 LLM)")
    parser.add_argument("--limit", type=int, default=40, help="生成条数 (默认 40)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出文件 (默认 scripts/eval_qa.jsonl)")
    args = parser.parse_args()

    chunks = {c["chunk_id"]: c for c in load_jsonl(CHUNKS_PROCESSED_PATH)}
    ent_rows = load_jsonl(ENTITIES_EXTRACTED_PATH)
    print(f"📂 chunk 数: {len(chunks)}, 实体行: {len(ent_rows)}")
    if not os.path.exists(ENTITIES_EXTRACTED_PATH):
        print(f"❌ 实体文件不存在: {ENTITIES_EXTRACTED_PATH}\n   请先运行 ETL 流水线提取实体")
        sys.exit(1)

    candidates = []
    seen = set()
    for row in ent_rows:
        cid = row.get("chunk_id")
        chunk = chunks.get(cid)
        if not chunk:
            continue
        content = (chunk.get("raw_content") or chunk.get("content") or "").strip()
        if len(content) < 40:
            continue
        entities = row.get("entities", [])
        q = gen_for_chunk(entities)
        if not q or q in seen:
            continue
        # 回标校验: 至少一个关键实体词出现在 chunk 原文中 (防实体与 chunk 错位)
        key_terms = [e["name"] for e in entities if e.get("type") in KEY_TYPES]
        if key_terms and not any(t in content for t in key_terms):
            continue
        seen.add(q)
        candidates.append({"question": q, "golden": cid})

    total = len(candidates)
    if total > args.limit:
        step = (total - 1) / (args.limit - 1) if args.limit > 1 else 0
        cands = [candidates[int(round(i * step))] for i in range(args.limit)]
    else:
        cands = candidates

    records = [
        {
            "id": i,
            "question": c["question"],
            "answer": "",
            "golden_chunk_ids": [c["golden"]],
            "intent": "fact",
        }
        for i, c in enumerate(cands, 1)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 生成 {len(records)} 条模板 QA → {args.out}")
    print(f"   候选 {total} 条 → 采样 {len(cands)} 条")
    print("   示例:")
    for r in records[:3]:
        print(f"   · {r['question']}  [{r['golden_chunk_ids'][0]}]")
    print(f"\n下一步: python scripts/eval_retrieval.py")


if __name__ == "__main__":
    main()
