"""
知识库重复检测 (只读诊断, 不改任何东西)

两层:
  1. 精确重复 —— content_hash 完全相同 (build_finance_kb 写入)
  2. 近似重复 —— 索引进里向量的两两余弦超阈值

第 2 层是有必要的: 精确哈希只能抓逐字节相同, 抓不到"同一主题换种说法"。
典型例子是 kb_risk_level 的「银行理财产品风险等级说明」与 risk_level.md 既有的
「产品风险等级 R1 到 R5」——讲的是同一件事, 只能靠向量相似度发现。

用法:
  .venv/Scripts/python.exe scripts/check_duplicates.py
  .venv/Scripts/python.exe scripts/check_duplicates.py --threshold 0.88 --top 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_index():
    import faiss
    from src.infra.paths import CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR

    index = faiss.read_index(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    return index, meta, chunks


def report_exact(chunks: list) -> None:
    groups = defaultdict(list)
    for c in chunks:
        h = c.get("content_hash")
        if h:
            groups[h].append(c)
    dupes = {h: cs for h, cs in groups.items() if len(cs) > 1}
    print(f"\n【精确重复】content_hash 相同的组: {len(dupes)}")
    for h, cs in dupes.items():
        print(f"  {h[:12]} × {len(cs)}")
        for c in cs:
            print(f"     {c['chunk_id']}  {c['source']}  {c.get('section', '')}")


def report_near(chunks: list, index, threshold: float, top: int) -> int:
    import numpy as np

    n = min(index.ntotal, len(chunks))
    vecs = np.vstack([index.reconstruct(i) for i in range(n)]).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1.0, norms)      # 保险: 索引里本已归一化

    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0.0)
    iu = np.triu_indices(n, k=1)
    scores = sim[iu]

    order = np.argsort(-scores)
    pairs = []
    for k in order[: max(top * 4, 200)]:
        s = float(scores[k])
        if s < threshold:
            break
        i, j = int(iu[0][k]), int(iu[1][k])
        pairs.append((s, i, j))

    print(f"\n【近似重复】余弦 ≥ {threshold} 的 chunk 对: {len(pairs)}")
    if not pairs:
        print("  无 —— 未发现同主题重复")
    for s, i, j in pairs[:top]:
        a, b = chunks[i], chunks[j]
        print(f"\n  {s:.4f}  {a['chunk_id']}  ↔  {b['chunk_id']}")
        print(f"          A: {a['source']} · {a.get('section', '')} · {a['content'][:56]}...")
        print(f"          B: {b['source']} · {b.get('section', '')} · {b['content'][:56]}...")
    if len(pairs) > top:
        print(f"\n  ... 另有 {len(pairs) - top} 对未显示 (--top 调整)")
    return len(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.90, help="近似重复余弦阈值")
    ap.add_argument("--top", type=int, default=25, help="最多显示多少对")
    args = ap.parse_args()

    index, _meta, chunks = load_index()
    print(f"知识库: {len(chunks)} chunk, 索引 {index.ntotal} 向量")

    report_exact(chunks)
    report_near(chunks, index, args.threshold, args.top)

    print("\n提示: 近似重复不一定要删 —— 先判断是「同主题重复」还是「必要互补」。"
          "\n      确认重复后, 改源文件 (finance_kb/*.md) 再跑 scripts/sync_kb.py, 不要手改产物。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
