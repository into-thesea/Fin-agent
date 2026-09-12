"""
知识库同步: 源变了吗 → 重建 → 校验 → 灌图 → 回归 → 记版本

设计要点:
  1. 单一事实源: 只认 data/finance_kb/*.md + catalog.jsonl + kg_domain_triples.jsonl
     + 构建脚本本身。产物 (chunks_processed.jsonl / FAISS / BM25 / kg_triples.jsonl / Neo4j)
     一律不许手写, 必须由本脚本派生。
  2. 指纹驱动: 源内容 hash 没变就直接退出, 不做无谓重建。
  3. 可回滚: 重建前快照产物, 校验不过自动还原。
  4. 重建即回归: 跑 eval_finance 离线指标并与上一轮 diff, 指标跌了非零退出。

扩知识库的正确姿势 = 往 data/finance_kb/ 里丢 .md 文件, 然后跑这个脚本。
不要再写脚本往 chunks_processed.jsonl 里 append —— 那是产物, 会被覆盖。

用法:
  .venv/Scripts/python.exe scripts/sync_kb.py             # 有变化才重建
  .venv/Scripts/python.exe scripts/sync_kb.py --force     # 强制重建
  .venv/Scripts/python.exe scripts/sync_kb.py --no-eval   # 跳过回归
退出码: 0 成功 / 1 校验失败已回滚 / 2 重建成功但回归指标变差
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:  # Windows 控制台默认 GBK, 中文日志会乱码
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KB_DIR = os.path.join(ROOT, "data", "finance_kb")
CATALOG = os.path.join(KB_DIR, "catalog.jsonl")
KG_DOMAIN = os.path.join(KB_DIR, "kg_domain_triples.jsonl")
MANIFEST = os.path.join(ROOT, "data", ".kb_manifest.json")
BACKUP_ROOT = os.path.join(ROOT, "data", "_backup_kb_sync")
BUILDER = os.path.join(ROOT, "scripts", "build_finance_kb.py")
HISTORY = os.path.join(ROOT, "data", "eval", "history.jsonl")

OFFLINE_KEYS = ("evidence_coverage", "path_structure_consistency", "graph_trigger_ratio")


# ── 指纹 ──────────────────────────────────────────────────────────

def source_files() -> list[str]:
    """唯一事实源清单 (与 build_finance_kb.collect_chunks 的跳过规则保持一致)

    除知识库内容外, 还把构建脚本本身算进来: 它决定派生结果怎么渲染,
    改了它等价于改了源 (产品文档的模板就在里面)。
    """
    files = sorted(f for f in glob.glob(os.path.join(KB_DIR, "*.md"))
                   if not os.path.basename(f).startswith("_"))
    files += [p for p in (CATALOG, KG_DOMAIN, BUILDER) if os.path.exists(p)]
    return files


def fingerprint() -> dict[str, str]:
    fp = {}
    for f in source_files():
        with open(f, "rb") as fh:
            key = os.path.relpath(f, ROOT).replace("\\", "/")
            fp[key] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return fp


def compute_version(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:12]


def load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def diff_keys(old: dict, new: dict) -> tuple[list, list, list]:
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    modified = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    return added, removed, modified


def _read_chunk_hashes() -> dict:
    """读当前产物的 chunk_id → content_hash"""
    from src.infra.paths import CHUNKS_PROCESSED_PATH
    out = {}
    try:
        with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    if c.get("chunk_id"):
                        out[c["chunk_id"]] = c.get("content_hash", "")
    except OSError:
        pass
    return out


def report_chunk_changes(old_chunks: dict, new_chunks: dict) -> None:
    """报出这次更新新增/修改/删除了哪些知识块

    这是"某个知识的更新以什么为依据"的最小可用形态: chunk_id 稳定 (按小节标题哈希)
    且带 content_hash, 所以能区分「新增的知识」和「旧知识被改了」——
    纯位置 ID 下这两种情况都表现为一片 id 位移, 分辨不出来。
    """
    added = sorted(set(new_chunks) - set(old_chunks))
    removed = sorted(set(old_chunks) - set(new_chunks))
    modified = sorted(k for k in set(old_chunks) & set(new_chunks)
                      if old_chunks[k] != new_chunks[k])
    if not (added or removed or modified):
        print("知识块无变化")
        return
    print(f"知识块变化: 新增 {len(added)} / 修改 {len(modified)} / 删除 {len(removed)}")
    for label, items in (("新增", added), ("修改", modified), ("删除", removed)):
        for cid in items[:4]:
            print(f"  {label}: {cid}")
        if len(items) > 4:
            print(f"  {label}: ... 另有 {len(items) - 4} 条")


# ── 快照 / 回滚 ───────────────────────────────────────────────────

def _artifacts():
    from src.infra.paths import BM25_INDEX_PATH, CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR
    return CHUNKS_PROCESSED_PATH, FAISS_INDEX_DIR, BM25_INDEX_PATH


def snapshot() -> str:
    chunks_p, faiss_d, bm25_p = _artifacts()
    dst = os.path.join(BACKUP_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    for p in (chunks_p, bm25_p):
        if os.path.exists(p):
            shutil.copy2(p, dst)
    if os.path.isdir(faiss_d):
        shutil.copytree(faiss_d, os.path.join(dst, "faiss_index"))
    return dst


def restore(src: str) -> None:
    chunks_p, faiss_d, bm25_p = _artifacts()
    for p in (chunks_p, bm25_p):
        b = os.path.join(src, os.path.basename(p))
        if os.path.exists(b):
            shutil.copy2(b, p)
    b = os.path.join(src, "faiss_index")
    if os.path.isdir(b):
        shutil.rmtree(faiss_d, ignore_errors=True)
        shutil.copytree(b, faiss_d)


# ── 校验 ──────────────────────────────────────────────────────────

def verify(n_chunks: int) -> tuple[bool, str]:
    """三路索引条数必须一致, 否则视为重建失败"""
    import faiss
    _, faiss_d, bm25_p = _artifacts()

    with open(os.path.join(faiss_d, "metadata.json"), "r", encoding="utf-8") as f:
        n_meta = len(json.load(f))
    n_vec = faiss.read_index(os.path.join(faiss_d, "index.faiss")).ntotal
    bm25_json = os.path.splitext(bm25_p)[0] + ".json"
    with open(bm25_json, "r", encoding="utf-8") as f:
        n_bm25 = json.load(f).get("total_docs", 0)

    detail = f"chunks={n_chunks} faiss={n_vec} metadata={n_meta} bm25={n_bm25}"
    return (n_chunks == n_vec == n_meta == n_bm25 and n_chunks > 0), detail


# ── 灌图 / 回归 ───────────────────────────────────────────────────

def reload_graph() -> None:
    """kg_triples.jsonl 变了才灌 Neo4j; Neo4j 没起就跳过(不阻塞本地索引)

    load_kg_to_neo4j.main() 连不上时调 sys.exit(1), 抛的是 SystemExit 而非 Exception,
    所以这里必须显式接住, 否则会把整个同步流程带崩。
    """
    try:
        import load_kg_to_neo4j
        load_kg_to_neo4j.main()
    except SystemExit as e:
        print(f"  Neo4j 灌库跳过 (exit={e.code}); 起库后重跑: docker compose up -d neo4j")
    except Exception as e:
        print(f"  Neo4j 灌库跳过: {type(e).__name__}: {e}")


def run_eval(version: str) -> tuple[dict, bool]:
    import eval_finance

    rows = eval_finance.load_golden()
    report = {"golden_n": len(rows), "version": version,
              "offline": eval_finance.offline_metrics(rows)}

    baseline = {}
    if os.path.exists(eval_finance.REPORT):
        try:
            with open(eval_finance.REPORT, "r", encoding="utf-8") as f:
                baseline = json.load(f)
        except Exception:
            pass

    with open(eval_finance.REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    o = report["offline"]
    print(f"\n回归: Golden={o['n']} 证据覆盖={o['evidence_coverage']} "
          f"路径结构一致={o['path_structure_consistency']} 图谱触发={o['graph_trigger_ratio']}")
    eval_finance.diff_against_baseline(baseline, report)

    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    summary = {k: v for k, v in o.items() if k != "rows"}  # 明细留在 report, 历史只留汇总量
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"),
                            "version": version, **summary}, ensure_ascii=False) + "\n")

    old = baseline.get("offline") or {}
    dropped = [k for k in OFFLINE_KEYS
               if isinstance(old.get(k), (int, float)) and isinstance(o.get(k), (int, float))
               and o[k] < old[k]]
    if dropped:
        print(f"  ⚠️  指标变差: {', '.join(dropped)}")
    return report, not dropped


# ── 主流程 ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="源没变也重建")
    ap.add_argument("--no-eval", action="store_true", help="跳过回归")
    args = ap.parse_args()

    fp = fingerprint()
    version = compute_version(fp)
    old = load_manifest()
    added, removed, modified = diff_keys(old.get("sources", {}), fp)

    if not (added or removed or modified) and not args.force:
        print(f"源无变化 (version={old.get('version', '?')}), 跳过重建")
        if "chunks" not in old:        # 老 manifest 回填一次快照, 否则下次变更报不出增量
            current = _read_chunk_hashes()
            if current:
                old["chunks"] = current
                with open(MANIFEST, "w", encoding="utf-8") as f:
                    json.dump(old, f, ensure_ascii=False, indent=2)
                print(f"已回填知识块快照 ({len(current)} 条)")
        return 0

    print(f"检测到源变化 → 重建 (新版本 {version})")
    for label, items in (("新增", added), ("删除", removed), ("修改", modified)):
        for it in items:
            print(f"  {label}: {it}")

    backup = snapshot()
    print(f"已快照 → {os.path.relpath(backup, ROOT)}")

    import build_finance_kb
    try:
        n_chunks = build_finance_kb.build_all(skip_milvus=False)
    except Exception as e:
        print(f"❌ 构建失败: {type(e).__name__}: {e} → 回滚")
        restore(backup)
        print("已回滚到重建前状态")
        return 1
    print(f"重建完成: {n_chunks} chunk")

    ok, detail = verify(n_chunks)
    if not ok:
        print(f"❌ 校验失败 ({detail}) → 回滚")
        restore(backup)
        print("已回滚到重建前状态")
        return 1
    print(f"✅ 校验通过 ({detail})")

    # 三元组现在由 catalog 派生, 每次重建都会重写, 所以直接灌图 (MERGE 幂等)
    print("灌 Neo4j")
    reload_graph()

    eval_ok = True
    if not args.no_eval:
        _, eval_ok = run_eval(version)

    new_chunks = _read_chunk_hashes()
    report_chunk_changes(old.get("chunks", {}), new_chunks)

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"version": version,
                   "synced_at": datetime.now().isoformat(timespec="seconds"),
                   "chunk_count": n_chunks,
                   "sources": fp,
                   "chunks": new_chunks}, f, ensure_ascii=False, indent=2)

    print(f"\n同步完成 · version={version} · chunks={n_chunks}")
    if not eval_ok:
        print("⚠️  回归指标有下降, 建议查看 data/eval/report_finance.json")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
