"""
Badcase 自动聚类分诊

用 BGE embedding 对 badcase 问题做向量化，按余弦相似度聚类，
自动标注类型（检索错/意图错/生成幻觉/工具失败/低置信度），
按影响范围排序输出。

用法:
  .venv/Scripts/python.exe scripts/badcase_cluster.py
  .venv/Scripts/python.exe scripts/badcase_cluster.py --threshold 0.75
  .venv/Scripts/python.exe scripts/badcase_cluster.py --top 50
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_badcases(limit: int = 100) -> list[dict]:
    """从 turn_traces 表加载 badcase 候选（SQL谓词派生的5类）"""
    from src.database import db_manager

    rows = db_manager.list_badcases(limit=limit)
    result = []
    for r in rows:
        reasons = []
        if r.get("intent") == "unknown":
            reasons.append("未知意图")
        if r.get("sources_json") == "[]":
            reasons.append("检索为空")
        review = r.get("review_json") or ""
        if '"verdict":"reject"' in review or '"verdict": "reject"' in review:
            reasons.append("审核被拒")
        if (r.get("confidence") or 1.0) < 0.5:
            reasons.append("低置信度")
        result.append({
            "trace_id": r.get("trace_id", ""),
            "query": r.get("query", ""),
            "intent": r.get("intent", ""),
            "reasons": reasons,
            "latency": r.get("latency", 0),
        })
    return result


def embed_queries(queries: list[str]) -> np.ndarray:
    """用 BGE 模型对问题做向量化"""
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()
    embeddings = retriever.model.encode(queries, normalize_embeddings=True)
    return np.array(embeddings)


def cluster_by_similarity(queries: list[str], embeddings: np.ndarray,
                           threshold: float = 0.75) -> list[list[int]]:
    """按余弦相似度聚类（贪心：和已有簇的质心相似度>阈值就归为一簇）"""
    clusters = []  # 每个元素是一个簇，包含query索引列表
    centroids = []  # 每个簇的质心向量

    for i, emb in enumerate(embeddings):
        assigned = False
        for ci, centroid in enumerate(centroids):
            sim = float(np.dot(emb, centroid))
            if sim >= threshold:
                clusters[ci].append(i)
                # 更新质心
                cluster_embs = embeddings[clusters[ci]]
                centroids[ci] = cluster_embs.mean(axis=0)
                centroids[ci] /= np.linalg.norm(centroids[ci])
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            centroids.append(emb.copy())

    return clusters


def auto_label_type(badcase: dict) -> str:
    """自动标注 badcase 类型"""
    reasons = badcase.get("reasons", [])
    if "检索为空" in reasons:
        return "检索失败"
    if "未知意图" in reasons:
        return "意图识别错误"
    if "审核被拒" in reasons:
        return "答案审核不通过（可能幻觉）"
    if "低置信度" in reasons:
        return "低置信度"
    return "其他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.75, help="聚类相似度阈值")
    ap.add_argument("--top", type=int, default=100, help="处理前N条badcase")
    ap.add_argument("--output", type=str, default="", help="输出JSON文件路径")
    args = ap.parse_args()

    print("加载 badcase 候选...")
    badcases = load_badcases(limit=args.top)
    print(f"共 {len(badcases)} 条 badcase")

    if not badcases:
        print("无 badcase 数据")
        return

    queries = [b["query"] for b in badcases]
    print(f"向量化 {len(queries)} 条问题...")
    embeddings = embed_queries(queries)

    print(f"聚类（阈值={args.threshold}）...")
    clusters = cluster_by_similarity(queries, embeddings, threshold=args.threshold)

    # 按簇大小排序
    clusters.sort(key=lambda c: len(c), reverse=True)

    print(f"\n{'='*60}")
    print(f"Badcase 聚类分诊结果：{len(badcases)} 条 → {len(clusters)} 个簇")
    print(f"{'='*60}\n")

    type_stats = {}
    output_clusters = []

    for ci, cluster in enumerate(clusters):
        cluster_badcases = [badcases[i] for i in cluster]
        # 簇代表问题（取第一条）
        representative = cluster_badcases[0]["query"]
        # 类型统计
        types = [auto_label_type(b) for b in cluster_badcases]
        dominant_type = max(set(types), key=types.count)
        # 原因统计
        all_reasons = []
        for b in cluster_badcases:
            all_reasons.extend(b["reasons"])
        reason_counts = {}
        for r in all_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

        for t in types:
            type_stats[t] = type_stats.get(t, 0) + 1

        print(f"【簇 {ci+1}】{len(cluster)} 条 · 类型: {dominant_type}")
        print(f"  代表问题: {representative[:60]}")
        print(f"  原因分布: {', '.join(f'{k}×{v}' for k, v in reason_counts.items())}")
        if len(cluster) <= 5:
            for b in cluster_badcases:
                print(f"    - {b['query'][:50]} (trace: {b['trace_id'][:12]})")
        else:
            for b in cluster_badcases[:3]:
                print(f"    - {b['query'][:50]} (trace: {b['trace_id'][:12]})")
            print(f"    ... 另有 {len(cluster)-3} 条")
        print()

        output_clusters.append({
            "cluster_id": ci + 1,
            "size": len(cluster),
            "dominant_type": dominant_type,
            "representative": representative,
            "reason_counts": reason_counts,
            "trace_ids": [b["trace_id"] for b in cluster_badcases],
            "queries": [b["query"] for b in cluster_badcases],
        })

    print(f"{'='*60}")
    print("类型统计:")
    for t, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t}: {count} 条 ({count/len(badcases)*100:.1f}%)")
    print(f"{'='*60}")

    if args.output:
        out_path = os.path.join(ROOT, args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "total": len(badcases),
                "cluster_count": len(clusters),
                "threshold": args.threshold,
                "type_stats": type_stats,
                "clusters": output_clusters,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
