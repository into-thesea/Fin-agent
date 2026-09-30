"""
Golden QA 绑定知识库版本

给每条 Golden QA 增加 kb_version 字段，记录该用例创建/验证时的知识库版本。
知识库更新后可检测：evidence 文件是否还存在、内容是否变化、答案是否需要复核。

用法:
  .venv/Scripts/python.exe scripts/golden_bind_version.py           # 给所有Golden绑定当前版本
  .venv/Scripts/python.exe scripts/golden_bind_version.py --check   # 检查知识库更新后哪些Golden需要复核
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GOLDEN = os.path.join(ROOT, "data", "eval", "finance_qa_golden.jsonl")
MANIFEST = os.path.join(ROOT, "data", ".kb_manifest.json")
KB_DIR = os.path.join(ROOT, "data", "finance_kb")


def get_current_version() -> str:
    """获取当前知识库版本号"""
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, "r", encoding="utf-8") as f:
                return json.load(f).get("version", "unknown")
        except Exception:
            pass
    return "unknown"


def get_file_hash(filepath: str) -> str:
    """计算文件的SHA256哈希（前16位）"""
    if not os.path.exists(filepath):
        return "MISSING"
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def bind_version():
    """给所有Golden QA绑定当前知识库版本"""
    version = get_current_version()
    rows = []
    with open(GOLDEN, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                # 记录evidence文件的当前哈希，用于后续检测变化
                evidence_hashes = {}
                for ev in row.get("evidence", []):
                    ev_path = os.path.join(KB_DIR, ev)
                    evidence_hashes[ev] = get_file_hash(ev_path)
                row["kb_version"] = version
                row["evidence_hashes"] = evidence_hashes
                rows.append(row)

    with open(GOLDEN, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已为 {len(rows)} 条 Golden QA 绑定知识库版本: {version}")
    need_review = sum(1 for r in rows if any(v == "MISSING" for v in r.get("evidence_hashes", {}).values()))
    if need_review:
        print(f"⚠️  {need_review} 条 Golden 的 evidence 文件缺失，需要复核")


def check_staleness():
    """检查知识库更新后哪些Golden需要复核（evidence文件变化或缺失）"""
    version = get_current_version()
    stale = []
    missing = []
    total = 0

    with open(GOLDEN, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                total += 1
                old_version = row.get("kb_version", "unknown")
                old_hashes = row.get("evidence_hashes", {})

                # 检查evidence文件是否变化
                changed_files = []
                missing_files = []
                for ev, old_hash in old_hashes.items():
                    ev_path = os.path.join(KB_DIR, ev)
                    current_hash = get_file_hash(ev_path)
                    if current_hash == "MISSING":
                        missing_files.append(ev)
                    elif current_hash != old_hash:
                        changed_files.append(ev)

                if missing_files:
                    missing.append({"id": row["id"], "question": row["question"][:40],
                                    "missing": missing_files, "old_version": old_version})
                elif changed_files and old_version != version:
                    stale.append({"id": row["id"], "question": row["question"][:40],
                                  "changed": changed_files, "old_version": old_version,
                                  "current_version": version})

    print(f"\n知识库版本: {version}")
    print(f"Golden QA 总数: {total}")
    print(f"\n需要复核（evidence文件已变化）: {len(stale)} 条")
    for s in stale[:10]:
        print(f"  [{s['id']}] {s['question']}")
        print(f"    变化文件: {', '.join(s['changed'])}")
        print(f"    版本: {s['old_version']} → {s['current_version']}")
    if len(stale) > 10:
        print(f"  ... 另有 {len(stale)-10} 条")

    print(f"\n需要修复（evidence文件已缺失）: {len(missing)} 条")
    for m in missing[:10]:
        print(f"  [{m['id']}] {m['question']}")
        print(f"    缺失文件: {', '.join(m['missing'])}")
    if len(missing) > 10:
        print(f"  ... 另有 {len(missing)-10} 条")

    if not stale and not missing:
        print("\n✅ 所有 Golden QA 的 evidence 均有效，无需复核")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="检查知识库更新后哪些Golden需要复核")
    args = ap.parse_args()

    if args.check:
        check_staleness()
    else:
        bind_version()


if __name__ == "__main__":
    main()
