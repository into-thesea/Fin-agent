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
        # 去掉 _cN 后缀归一化为文档级 ID，与 golden_chunk_ids 格式对齐
        # 同一文档的不同 chunk 内容相关，任意 chunk 命中即算对
        return re.sub(r"_c\d+$", "", cid)
    key = re.sub(r"\s+", "", item.get("content") or "")[:60]
    return prefix_index.get(key)


def file_prefix(chunk_id):
    """文档级宽松匹配: prod_p002.md_0114 -> prod_p002.md"""
    if not chunk_id:
        return chunk_id
    head, _, tail = chunk_id.rpartition("_")
    return head if tail.isdigit() else chunk_id


def main():
    parser = argparse.ArgumentParser(description="RAG 检索召回率评测")
    parser.add_argument("--eval", default=DEFAULT_EVAL, help="评测集路径 (默认 scripts/eval_qa.jsonl)")
    parser.add_argument("--max-k", type=int, default=10, help="检索 top-K 上限 (默认 10)")
    parser.add_argument("--file-level", action="store_true", help="文档级宽松匹配 (同文档任意 chunk 命中即算, 用于诊断 golden 不唯一)")
    parser.add_argument("--by-evidence", action="store_true",
                        help="用评测集的 evidence(文件名) 对齐检索结果的 source 字段。"
                             "理财 golden (data/eval/finance_qa_golden.jsonl) 用的是 evidence, "
                             "不指定时若记录里没有 golden_chunk_ids 会自动启用该模式")
    parser.add_argument("--rerank", action="store_true", help="启用重排 (本地有 bge-reranker-base 用 CrossEncoder, 否则 SparseBoost)")
    parser.add_argument("--rerank-type", choices=["auto", "api", "cross", "sparse"], default="auto",
                        help="重排器类型: api=DashScope语义精排(推荐) / cross=本地CrossEncoder / "
                             "sparse=自研SparseBoost(实测负收益,仅供复现) / auto=有CE模型用CE否则用api")
    parser.add_argument("--lambd", type=float, default=0.6, help="SparseBoost 重叠率权重 (默认 0.6)")
    parser.add_argument("--pool-k", type=int, default=0, help="候选池大小 (默认 0=auto: max(max_k*6,30); 调大提升覆盖率)")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条 (0=全部; 分批跑可防 CPU 长跑 segfault)")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条 (与 --limit 配合分批)")
    parser.add_argument("--out", default="", help="可选: 保存逐条结果 JSON")
    parser.add_argument("--with-rewrite", action="store_true",
                        help="先做 Query Rewriting 再检索，对比改写前后的 Recall@5 "
                             "(从 golden chunk 提取产品名模拟上一轮上下文)")
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

    # 对齐方式: golden_chunk_ids (chunk 级) 还是 evidence (文件名级)
    has_chunk_ids = any(r.get("golden_chunk_ids") for r in records)
    has_evidence = any(r.get("evidence") for r in records)
    use_evidence = args.by_evidence or (not has_chunk_ids and has_evidence)
    if use_evidence:
        print("🔗 对齐方式: evidence(文件名) ↔ 检索结果 source")
    if not use_evidence:
        chunk_map, prefix_index = build_chunk_map()
        missing = 0
        for r in records:
            for g in r.get("golden_chunk_ids") or []:
                if g not in chunk_map:
                    missing += 1
                    print(f"  ⚠️  golden chunk 不在分块文件中: {g}")
        if missing:
            print(f"   (以上 {missing} 个 golden 不在索引来源中, 对应问题的 Recall 可能偏低)")
    else:
        prefix_index = {}

    retriever = HybridRetriever()
    print("🔍 混合检索器就绪 (Milvus + BM25 + 图谱)\n")

    # Query Rewriting 相关初始化
    product_names = []
    rewrite_fn = None
    if args.with_rewrite:
        from src.core.query_rewriter import rewrite_query
        rewrite_fn = rewrite_query
        catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "data", "finance_kb", "catalog.jsonl")
        if os.path.exists(catalog_path):
            with open(catalog_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            p = json.loads(line)
                            if p.get("name"):
                                product_names.append(p["name"])
                        except json.JSONDecodeError:
                            continue
            print(f"🔄 Query Rewriting 已启用，加载产品名 {len(product_names)} 个\n")
        else:
            print(f"⚠️ 未找到 catalog.jsonl，Query Rewriting 无法提取产品名\n")
            args.with_rewrite = False

    # 可选精排: 扩大候选池 (pool_k) 后再收敛到 max_k
    reranker = None
    pool_k = max_k
    if args.rerank:
        pool_k = args.pool_k or max(max_k * 6, 30)
        ce_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models", "bge-reranker-base"
        )
        # rerank-type: api(DashScope 语义精排, 推荐) / cross(本地CrossEncoder) /
        #              sparse(自研 SparseBoost —— 实测负收益, 仅用于复现结论)
        want_cross = (args.rerank_type == "cross") or (
            args.rerank_type == "auto" and os.path.exists(ce_path)
        )
        if args.rerank_type == "api":
            from src.retrieval.reranker import ApiReranker
            reranker = ApiReranker()
            print(f"🔎 语义精排 (gte-rerank-v2): 候选池 {pool_k} → 精排取 top-{max_k}\n")
        elif want_cross:
            try:
                from src.retrieval.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker(ce_path)
                print(f"🔎 CrossEncoder 精排已启用 (两阶段: 池{pool_k}→粗排30→精排{max_k})\n")
            except Exception as e:
                print(f"⚠️ CrossEncoder 加载失败, 改用语义精排: {e}")
                reranker = None
        if reranker is None:
            # 兜底用语义精排, 不用 SparseBoost —— 后者在理财域实测负收益(见 reranker.py)
            if args.rerank_type == "sparse":
                from src.retrieval.reranker import SparseReranker
                reranker = SparseReranker(lambd_overlap=args.lambd)
                print(f"🔎 SparseBoost 精排 (lambd={args.lambd}, 实测负收益, 仅供复现): "
                      f"候选池 {pool_k} → 精排取 top-{max_k}\n")
            else:
                from src.retrieval.reranker import ApiReranker
                reranker = ApiReranker()
                print(f"🔎 语义精排 (gte-rerank-v2): 候选池 {pool_k} → 精排取 top-{max_k}\n")

    hits = {k: 0 for k in range(1, max_k + 1)}
    mrr_total = 0.0
    no_chunk_id = 0
    per_record = []
    # Query Rewriting 对比统计
    hits_orig = {k: 0 for k in range(1, max_k + 1)}
    hits_rewritten = {k: 0 for k in range(1, max_k + 1)}
    mrr_orig = 0.0
    mrr_rewritten = 0.0
    rewrite_count = 0
    rewrite_improved = 0
    rewrite_downgraded = 0

    def key_of(item):
        """把一个检索结果映射成用于比对 golden 的键"""
        if use_evidence:
            return item.get("source") or None
        cid = chunk_key(item, prefix_index)
        if cid is None:
            return None
        return file_prefix(cid) if args.file_level else cid

    def _retrieve_and_score(query, golden_set):
        """检索并计算命中排名，返回 (rank, retrieved, error)"""
        try:
            result = retriever.hybrid_retrieve(query, top_k=pool_k)
        except Exception as e:
            return None, [], str(e)
        retrieved = result.get("local", [])
        if reranker is not None:
            if type(reranker).__name__ == "CrossEncoderReranker":
                from src.retrieval.reranker import SparseReranker
                coarse = SparseReranker(lambd_overlap=args.lambd).rerank(query, retrieved, top_k=30)
                retrieved = reranker.rerank(query, coarse, top_k=max_k)
            else:
                retrieved = reranker.rerank(query, retrieved, top_k=max_k)
        else:
            retrieved = retrieved[:max_k]
        rank = None
        for k in range(1, max_k + 1):
            ids = set()
            for item in retrieved[:k]:
                key = key_of(item)
                if key is None:
                    continue
                ids.add(key)
            if ids & golden_set:
                rank = k
                break
        return rank, retrieved, None

    for i, rec in enumerate(records, 1):
        q = rec["question"]
        if use_evidence:
            golden = set(rec.get("evidence") or [])
        else:
            golden = set(rec.get("golden_chunk_ids") or [])
            golden = {re.sub(r"_c\d+$", "", g) for g in golden}
            if args.file_level:
                golden = {file_prefix(g) for g in golden}

        # Query Rewriting: 优先用记录中的 context_product（模拟上一轮上下文），
        # 没有则从 golden chunk 内容中匹配产品名作为兜底
        q_rewritten = q
        if args.with_rewrite and rewrite_fn and not use_evidence:
            matched_product = rec.get("context_product")
            if not matched_product and product_names:
                golden_content = ""
                for gid in rec.get("golden_chunk_ids", []):
                    if gid in chunk_map:
                        golden_content += chunk_map[gid] + " "
                for pname in product_names:
                    if pname in golden_content:
                        matched_product = pname
                        break
            if matched_product:
                q_rewritten = rewrite_fn(q, {"product": matched_product}, use_llm=False)
                if q_rewritten != q:
                    rewrite_count += 1

        # 检索（改写后）
        rank, retrieved, err = _retrieve_and_score(q_rewritten, golden)
        if err:
            print(f"  ❌ 检索失败 [{q[:40]}]: {err}")
            continue
        if rank:
            for k in range(rank, max_k + 1):
                hits[k] += 1
            mrr_total += 1.0 / rank
            if q_rewritten != q:
                for k in range(rank, max_k + 1):
                    hits_rewritten[k] += 1
                mrr_rewritten += 1.0 / rank
        else:
            print(f"  ⚠️  未命中: {q[:50]}" + (f" → 改写: {q_rewritten[:30]}" if q_rewritten != q else ""))

        # 对比：原始 query 的检索结果（仅统计，不打印未命中）
        if args.with_rewrite and q_rewritten != q:
            rank_orig, _, _ = _retrieve_and_score(q, golden)
            if rank_orig:
                for k in range(rank_orig, max_k + 1):
                    hits_orig[k] += 1
                mrr_orig += 1.0 / rank_orig
            if rank and not rank_orig:
                rewrite_improved += 1
            elif not rank and rank_orig:
                rewrite_downgraded += 1

        per_record.append({
            "id": rec.get("id"),
            "question": q,
            "rewritten": q_rewritten if q_rewritten != q else None,
            "golden": sorted(golden),
            "retrieved": [key_of(x) for x in retrieved],
            "hit_rank": rank,
        })
        if i % 10 == 0 or i == len(records):
            print(f"  进度 {i}/{len(records)}")

    n = len(records)
    mode_label = " (文档级宽松)" if args.file_level else " (chunk 级严格)"
    if args.with_rewrite:
        mode_label += " + Query Rewriting"
    if args.rerank:
        _name = type(reranker).__name__ if reranker is not None else "None"
        mode_label += {
            "ApiReranker": " + 语义精排(gte-rerank-v2)",
            "CrossEncoderReranker": " + CrossEncoder",
            "SparseReranker": f" + SparseBoost(lambd={args.lambd})",
        }.get(_name, f" + {_name}")
    print("\n" + "=" * 46)
    print(f"📊 检索召回率报告{mode_label}")
    print("=" * 46)
    for k in range(1, max_k + 1):
        bar = "#" * max(0, int(hits[k] / n * 30))
        print(f"  Recall@{k:<2d} = {hits[k]/n:6.1%}  {bar}")
    print(f"  MRR       = {mrr_total/n:.4f}")
    print("=" * 46)

    # Query Rewriting 对比输出
    if args.with_rewrite and rewrite_count > 0:
        print(f"\n🔄 Query Rewriting 对比（{rewrite_count} 条问题被改写）:")
        print(f"  {'指标':<12} {'改写前':<12} {'改写后':<12} {'变化':<10}")
        for k in [1, 3, 5]:
            if k <= max_k:
                orig = hits_orig[k] / rewrite_count if rewrite_count else 0
                new = hits_rewritten[k] / rewrite_count if rewrite_count else 0
                delta = new - orig
                print(f"  Recall@{k:<3d}    {orig:6.1%}        {new:6.1%}        {delta:+.1%}")
        orig_mrr = mrr_orig / rewrite_count if rewrite_count else 0
        new_mrr = mrr_rewritten / rewrite_count if rewrite_count else 0
        print(f"  {'MRR':<12} {orig_mrr:.4f}        {new_mrr:.4f}        {new_mrr-orig_mrr:+.4f}")
        print(f"\n  改写后提升命中: {rewrite_improved} 条 (原未命中→改写后命中)")
        print(f"  改写后降低命中: {rewrite_downgraded} 条 (原命中→改写后未命中)")
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
