"""
问答检索召回率评测脚本 — Recall@K + MRR

对 eval_qa.jsonl 中的每个问题执行四路混合检索 (FAISS + BM25 + 图谱 + 社区)，
用 golden_chunk_ids 判定 top-K 是否命中了答案所在 chunk。

指标:
  Recall@K : 正确答案 chunk 出现在前 K 条检索结果中的问题占比
  MRR      : 第一个正确答案 chunk 排名倒数的均值 (越小越早命中)

用法:
    python scripts/eval_retrieval.py                 # 全量评测, Recall@1..10
    python scripts/eval_retrieval.py --max-k 5 --out scripts/eval_results.json
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

from src.infra.paths import CHUNKS_PROCESSED_PATH
from src.retrieval.retriever import HybridRetriever

DEFAULT_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_qa.jsonl")


def load_records(path):
    if not os.path.exists(path):
        print(f"❌ 评测集不存在: {path}\n   请先运行 python scripts/generate_qa.py")
        sys.exit(1)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_chunk_map():
    """chunk_id -> content 映射, 同时构建 content 前缀反查索引 (兼容无 chunk_id 的旧索引)"""
    chunk_map = {}
    if os.path.exists(CHUNKS_PROCESSED_PATH):
        with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                    chunk_map[c["chunk_id"]] = c.get("raw_content") or c.get("content", "")
                except json.JSONDecodeError:
                    continue
    prefix_index = {}
    for cid, content in chunk_map.items():
        key = re.sub(r"\s+", "", content)[:60]
        prefix_index[key] = cid
    return chunk_map, prefix_index


def chunk_key(item, prefix_index):
    """提取检索结果的稳定 chunk 标识: 优先 chunk_id, 否则用 content 前缀反查"""
    cid = item.get("chunk_id")
    if cid:
        return cid
    key = re.sub(r"\s+", "", item.get("content") or "")[:60]
    return prefix_index.get(key)


def file_prefix(chunk_id):
    """文档级宽松匹配: BYD_2024_Annual_Report.pdf_0114 -> BYD_2024_Annual_Report.pdf"""
    if not chunk_id:
        return chunk_id
    head, _, tail = chunk_id.rpartition("_")
    return head if tail.isdigit() else chunk_id


def main():
    parser = argparse.ArgumentParser(description="RAG 检索召回率评测")
    parser.add_argument("--eval", default=DEFAULT_EVAL, help="评测集路径 (默认 scripts/eval_qa.jsonl)")
    parser.add_argument("--max-k", type=int, default=10, help="检索 top-K 上限 (默认 10)")
    parser.add_argument("--file-level", action="store_true", help="文档级宽松匹配 (同文档任意 chunk 命中即算, 用于诊断 golden 不唯一)")
    parser.add_argument("--rerank", action="store_true", help="启用重排 (本地有 bge-reranker-base 用 CrossEncoder, 否则 SparseBoost)")
    parser.add_argument("--rerank-type", choices=["auto", "cross", "sparse"], default="auto", help="重排器类型 (默认 auto; cross=强制CrossEncoder, sparse=强制SparseBoost)")
    parser.add_argument("--lambd", type=float, default=0.6, help="SparseBoost 重叠率权重 (默认 0.6)")
    parser.add_argument("--pool-k", type=int, default=0, help="候选池大小 (默认 0=auto: max(max_k*6,30); 调大提升覆盖率)")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条 (0=全部; 分批跑可防 CPU 长跑 segfault)")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条 (与 --limit 配合分批)")
    parser.add_argument("--out", default="", help="可选: 保存逐条结果 JSON")
    args = parser.parse_args()

    records = load_records(args.eval)
    max_k = args.max_k
    if args.offset > 0:
        records = records[args.offset:]
    if args.limit > 0:
        records = records[:args.limit]
    if not records:
        print("❌ 评测集为空")
        sys.exit(1)
    scope = f" (第 {args.offset+1}-{args.offset+len(records)} 条)" if (args.offset or args.limit) else ""
    print(f"📋 评测集: {len(records)} 条问题{scope}")

    chunk_map, prefix_index = build_chunk_map()
    missing = 0
    for r in records:
        for g in r.get("golden_chunk_ids") or []:
            if g not in chunk_map:
                missing += 1
                print(f"  ⚠️  golden chunk 不在分块文件中: {g}")
    if missing:
        print(f"   (以上 {missing} 个 golden 不在索引来源中, 对应问题的 Recall 可能偏低)")

    retriever = HybridRetriever()
    print("🔍 混合检索器就绪 (FAISS + BM25 + 图谱 + 社区)\n")

    # 可选 SparseBoost 重排: 扩大候选池 (pool_k) 后按稀疏信号精排
    reranker = None
    pool_k = max_k
    if args.rerank:
        from src.retrieval.reranker import SparseReranker
        pool_k = args.pool_k or max(max_k * 6, 30)
        ce_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models", "bge-reranker-base"
        )
        # rerank-type: auto(有模型用CE) / cross(强制CE) / sparse(强制SparseBoost)
        want_cross = (args.rerank_type == "cross") or (
            args.rerank_type == "auto" and os.path.exists(ce_path)
        )
        if want_cross:
            try:
                from src.retrieval.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker(ce_path)
                print(f"🔎 CrossEncoder 精排已启用 (两阶段: 池{pool_k}→粗排30→精排{max_k})\n")
            except Exception as e:
                print(f"⚠️ CrossEncoder 加载失败, 回退 SparseBoost: {e}")
                reranker = None
        if reranker is None:
            reranker = SparseReranker(lambd_overlap=args.lambd)
            print(f"🔎 SparseBoost 精排 (lambd={args.lambd}): 候选池 {pool_k} → 精排取 top-{max_k}\n")

    hits = {k: 0 for k in range(1, max_k + 1)}
    mrr_total = 0.0
    no_chunk_id = 0
    per_record = []

    for i, rec in enumerate(records, 1):
        q = rec["question"]
        golden = set(rec.get("golden_chunk_ids") or [])
        # 文档级宽松模式: golden 与检索结果都映射为文件前缀再比较
        if args.file_level:
            golden = {file_prefix(g) for g in golden}
        try:
            result = retriever.hybrid_retrieve(q, top_k=pool_k)
        except Exception as e:
            print(f"  ❌ 检索失败 [{q[:40]}]: {e}")
            continue
        retrieved = result.get("local", [])
        if reranker is not None:
            if type(reranker).__name__ == "CrossEncoderReranker":
                # 两阶段精排: SparseBoost 粗排大池到 30 (毫秒级), CrossEncoder 精排到 top-k
                from src.retrieval.reranker import SparseReranker
                coarse = SparseReranker(lambd_overlap=args.lambd).rerank(q, retrieved, top_k=30)
                retrieved = reranker.rerank(q, coarse, top_k=max_k)
            else:
                retrieved = reranker.rerank(q, retrieved, top_k=max_k)
        else:
            retrieved = retrieved[:max_k]

        # 命中判定: 前 k 条内是否出现 golden chunk
        rank = None
        for k in range(1, max_k + 1):
            ids = set()
            for item in retrieved[:k]:
                cid = chunk_key(item, prefix_index)
                if cid is None:
                    no_chunk_id += 1
                    continue
                ids.add(file_prefix(cid) if args.file_level else cid)
            if ids & golden:
                hits[k] += 1
                if rank is None:
                    rank = k
        if rank:
            mrr_total += 1.0 / rank
        else:
            print(f"  ⚠️  未命中: {q[:50]}")

        per_record.append({
            "id": rec.get("id"),
            "question": q,
            "golden": sorted(golden),
            "retrieved": [chunk_key(x, prefix_index) for x in retrieved],
            "hit_rank": rank,
        })
        if i % 10 == 0 or i == len(records):
            print(f"  进度 {i}/{len(records)}")

    n = len(records)
    mode_label = " (文档级宽松)" if args.file_level else " (chunk 级严格)"
    if args.rerank:
        if reranker is not None and type(reranker).__name__ == "CrossEncoderReranker":
            mode_label += " + CrossEncoder"
        else:
            mode_label += f" + SparseBoost(lambd={args.lambd})"
    print("\n" + "=" * 46)
    print(f"📊 检索召回率报告{mode_label}")
    print("=" * 46)
    for k in range(1, max_k + 1):
        bar = "#" * max(0, int(hits[k] / n * 30))
        print(f"  Recall@{k:<2d} = {hits[k]/n:6.1%}  {bar}")
    print(f"  MRR       = {mrr_total/n:.4f}")
    print("=" * 46)
    if no_chunk_id:
        print(f"  (提示: {no_chunk_id} 个检索结果缺少 chunk_id, 已用 content 前缀回查定位)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "total": n,
                "recall": {k: hits[k] / n for k in range(1, max_k + 1)},
                "mrr": mrr_total / n,
                "per_record": per_record,
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 逐条结果已保存: {args.out}")


if __name__ == "__main__":
    main()
