"""
一次性迁移 (b 方案第 1 步): 把 prod_p001-p011.md 的内容搬进 catalog.jsonl

背景:
  catalog.jsonl (结构化) 与 prod_pNNN.md (话术) 原本是两份手写事实源,
  其中 prod 的「产品要素」小节是 catalog 字段的纯重复 —— 一旦 catalog 改了而 .md 没改
  就产生漂移, 或者像 P012-P024 那样只加了一边。

做法:
  - 「产品要素」小节不再手写, 改由 build_finance_kb 从结构化字段模板生成;
    但该节末尾的免责句 (…以《产品说明书》…为准) 是产品差异化的, 抽到 elements_note 保留
  - 其余小节原样搬进 sections: [{title, body}]
  - source 标识仍是 prod_pNNN.md, 保证 golden 的 evidence 颗粒度不变

幂等: 已是 sections 结构的条目直接跳过。可重复运行。

用法: .venv/Scripts/python.exe scripts/_migrate_prod_docs_to_catalog.py [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "data", "finance_kb")
CATALOG = os.path.join(KB_DIR, "catalog.jsonl")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_sections(md_path: str) -> list[dict]:
    """按 '## ' 切分, 返回 [{title, body}]"""
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    out, title, body = [], None, []
    for line in lines:
        if line.startswith("## "):
            if title:
                out.append({"title": title, "body": "\n".join(body)})
            title, body = line[3:].strip(), []
        elif line.strip() and title is not None:
            body.append(line.strip())
    if title:
        out.append({"title": title, "body": "\n".join(body)})
    return out


def extract_elements_note(body: str) -> str:
    """从「产品要素」正文里抽出『…以《产品说明书》/发行公告为准』那句免责声明"""
    for sent in re.split(r"(?<=。)", body):
        if "为准" in sent and "产品要素" in sent:
            return sent.strip()
    return "产品要素与收益口径以《产品说明书》为准。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = []
    with open(CATALOG, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    by_id = {e["id"]: e for e in entries}

    migrated, skipped, missing = [], [], []
    for md in sorted(glob.glob(os.path.join(KB_DIR, "prod_p*.md"))):
        pid = os.path.splitext(os.path.basename(md))[0].replace("prod_", "").upper()
        e = by_id.get(pid)
        if not e:
            missing.append(pid)
            continue
        if "sections" in e:
            skipped.append(pid)
            continue

        secs = parse_sections(md)
        elements = next((s for s in secs if s["title"] == "产品要素"), None)
        e["elements_note"] = extract_elements_note(elements["body"] if elements else "")
        e["sections"] = [s for s in secs if s["title"] != "产品要素"]
        migrated.append(pid)

    print(f"迁移: {len(migrated)} 个 {migrated}")
    if skipped:
        print(f"跳过(已是 sections 结构): {len(skipped)} 个 {skipped}")
    if missing:
        print(f"⚠️  catalog 里找不到对应条目: {missing}")

    if args.dry_run:
        print("\n--dry-run, 未写入")
        for pid in migrated[:2]:
            e = by_id[pid]
            print(f"\n[{pid}] elements_note = {e['elements_note']}")
            print(f"      sections = {[s['title'] for s in e['sections']]}")
        return 0

    if migrated:
        bk = os.path.join(KB_DIR, f"catalog.bak_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        shutil.copy2(CATALOG, bk)
        print(f"已备份 → {os.path.relpath(bk, ROOT)}")
        with open(CATALOG, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"已写回 {CATALOG} ({len(entries)} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
