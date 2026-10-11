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
退出码: 0 成功 / 1 校验失败已回滚 / 2 重建成功但回归指标变差 / 3 索引重建成功但图谱未收敛
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

OFFLINE_KEYS = ("evidence_coverage", "chunk_info_hit_rate", "path_structure_consistency", "graph_trigger_ratio")
ONLINE_KEYS = ("intent_accuracy", "compliance_pass_rate")


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


def save_manifest(m: dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


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
    from src.infra.paths import BM25_INDEX_PATH, CHUNKS_PROCESSED_PATH
    return CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH


def snapshot() -> str:
    chunks_p, bm25_p = _artifacts()
    dst = os.path.join(BACKUP_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    for p in (chunks_p, bm25_p):
        if os.path.exists(p):
            shutil.copy2(p, dst)
    return dst


def restore(src: str) -> None:
    chunks_p, bm25_p = _artifacts()
    for p in (chunks_p, bm25_p):
        b = os.path.join(src, os.path.basename(p))
        if os.path.exists(b):
            shutil.copy2(b, p)


# ── 校验 ──────────────────────────────────────────────────────────

def _count_indexed(chunks_p: str) -> int:
    """分块文件里真正进索引的行数

    父子分块后文件里同时躺着父块和子块, 而父块**按设计不入检索索引**
    (vector_indexer 显式过滤 chunk_type == "parent")。拿文件总行数去跟
    Milvus/BM25 比, 会永远对不上 → 校验失败 → 回滚, sync_kb 一次都跑不完,
    manifest 停在旧版本、图谱也就跟着永远落后于源。
    """
    n = 0
    with open(chunks_p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line).get("chunk_type") != "parent":
                n += 1
    return n


def verify(n_chunks: int) -> tuple[bool, str]:
    """分块文件 / Milvus / BM25 三路条数必须一致, 否则视为重建失败

    快照/回滚只覆盖本地文件 (分块文件 + BM25); Milvus 是外部服务, 失败时
    由下一次全量重建覆盖, 这里只做一致性核对。
    """
    from src.vectorstore.milvus_manager import MilvusManager

    chunks_p, bm25_p = _artifacts()
    bm25_json = os.path.splitext(bm25_p)[0] + ".json"
    with open(bm25_json, "r", encoding="utf-8") as f:
        n_bm25 = json.load(f).get("total_docs", 0)

    mgr = MilvusManager()
    n_vec = mgr.total_count if mgr.available else -1

    n_indexed = _count_indexed(chunks_p)
    detail = (f"indexable={n_indexed}(文件 {n_chunks} 行, 父块 {n_chunks - n_indexed} 不入索引) "
              f"milvus={n_vec} bm25={n_bm25}")
    return (n_indexed == n_vec == n_bm25 and n_indexed > 0), detail


# ── 灌图 / 回归 ───────────────────────────────────────────────────

def reload_graph() -> bool:
    """把源派生出的 kg_triples.jsonl 收敛到 Neo4j, 返回是否成功

    Neo4j 没起不阻塞本地索引重建, 但**必须让失败可见**: 这里以前把灌库失败吞成一句
    "跳过", main() 照样返回 0, 结果 Neo4j 悄悄落后源 24 条边没人发现 —— main() 现在
    会非零退出, 且不把 graph_version 记进 manifest, 下次跑本脚本会自动补灌。

    load_kg_to_neo4j.main() 连不上时调 sys.exit(1), 抛的是 SystemExit 而非 Exception,
    所以必须显式接住, 否则会把整个同步流程带崩。
    """
    try:
        import load_kg_to_neo4j
    except Exception as e:
        print(f"  ❌ Neo4j 灌库失败: 无法导入 load_kg_to_neo4j: {e}")
        return False
    try:
        load_kg_to_neo4j.main()
        return True
    except SystemExit as e:
        print(f"  ❌ Neo4j 灌库失败 (exit={e.code}); 先 docker compose up -d neo4j 再重跑本脚本补灌")
        return False
    except Exception as e:
        print(f"  ❌ Neo4j 灌库失败: {type(e).__name__}: {e}")
        return False


def run_eval(version: str, with_answers: bool = False, online_limit: int = 0) -> tuple[dict, bool]:
    import eval_finance

    rows = eval_finance.load_golden()
    report = {"golden_n": len(rows), "version": version,
              "offline": eval_finance.offline_metrics(rows)}

    if with_answers:
        limit = online_limit or len(rows)
        report["online"] = eval_finance.answer_metrics(rows, limit)

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
    print(f"\n回归[离线]: Golden={o['n']} 证据覆盖={o['evidence_coverage']} "
          f"chunk信息命中={o.get('chunk_info_hit_rate')} "
          f"路径结构一致={o['path_structure_consistency']} 图谱触发={o['graph_trigger_ratio']}")
    if "online" in report:
        on = report["online"]
        print(f"回归[在线]: 意图准确率={on['intent_accuracy']} 合规通过率={on['compliance_pass_rate']}")
    eval_finance.diff_against_baseline(baseline, report)

    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    summary = {k: v for k, v in o.items() if k != "rows"}
    if "online" in report:
        summary.update({f"online_{k}": v for k, v in report["online"].items()})
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"),
                            "version": version, **summary}, ensure_ascii=False) + "\n")

    dropped = []
    old_offline = baseline.get("offline") or {}
    for k in OFFLINE_KEYS:
        if isinstance(old_offline.get(k), (int, float)) and isinstance(o.get(k), (int, float)) and o[k] < old_offline[k]:
            dropped.append(k)
    if "online" in report:
        old_online = baseline.get("online") or {}
        for k in ONLINE_KEYS:
            if isinstance(old_online.get(k), (int, float)) and isinstance(report["online"].get(k), (int, float)) and report["online"][k] < old_online[k]:
                dropped.append(f"online_{k}")
    if dropped:
        print(f"  ⚠️  指标变差: {', '.join(dropped)}")
    return report, not dropped


# ── 主流程 ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="源没变也重建")
    ap.add_argument("--no-eval", action="store_true", help="跳过回归")
    ap.add_argument("--with-answers", action="store_true", help="回归时跑在线评测（意图准确率+合规通过率，需LLM）")
    ap.add_argument("--online-limit", type=int, default=0, help="在线评测条数限制（0=全部）")
    args = ap.parse_args()

    fp = fingerprint()
    version = compute_version(fp)
    old = load_manifest()
    added, removed, modified = diff_keys(old.get("sources", {}), fp)

    if not (added or removed or modified) and not args.force:
        print(f"源无变化 (version={old.get('version', '?')}), 跳过重建")
        if old.get("graph_version") != version:
            # 上次灌图失败过 (manifest 里没记上这个版本) → 只补灌, 不重跑整条重建。
            # 少了这一步, 灌库跳过一次源又没再变, 图谱就永久落后, 只能靠人发现。
            print("上次图谱未收敛 → 只补灌 Neo4j")
            if not reload_graph():
                return 3
            old["graph_version"] = version
            save_manifest(old)
            print("✅ 图谱补灌完成")
            return 0
        if "chunks" not in old:        # 老 manifest 回填一次快照, 否则下次变更报不出增量
            current = _read_chunk_hashes()
            if current:
                old["chunks"] = current
                save_manifest(old)
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

    # 三元组由 catalog + kg_domain_triples 派生, 每次重建都会重写; 灌图是收敛式(补边+删陈旧边)
    print("灌 Neo4j")
    graph_ok = reload_graph()

    eval_ok = True
    if not args.no_eval:
        _, eval_ok = run_eval(version, with_answers=args.with_answers,
                               online_limit=args.online_limit)

    new_chunks = _read_chunk_hashes()
    report_chunk_changes(old.get("chunks", {}), new_chunks)

    manifest = {"version": version,
                "synced_at": datetime.now().isoformat(timespec="seconds"),
                "chunk_count": n_chunks,
                "sources": fp,
                "chunks": new_chunks}
    if graph_ok:
        manifest["graph_version"] = version   # 灌成功才记; 没记上就等于挂了个"下次补灌"的账
    save_manifest(manifest)

    print(f"\n同步完成 · version={version} · chunks={n_chunks}")
    if not graph_ok:
        print("❌ 图谱未收敛 —— 索引已更新但 Neo4j 落后于源; 起好 Neo4j 后重跑本脚本会自动补灌")
        return 3
    if not eval_ok:
        print("⚠️  回归指标有下降, 建议查看 data/eval/report_finance.json")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
