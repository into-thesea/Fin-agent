"""回填 documents 表里缺失的 content_hash

背景: 判重有两层, 都靠 content_hash —— 上传口的 find_document_by_hash, 和
add_document 里"同 filename 且同 hash 就跳过"。老记录的 hash 是 NULL 时**两层
一起失效**: 同一份 PDF 再传一次会被当成"新版本"插一条新记录, 内容在 Milvus/BM25
里存两份 (实测 6 份 PDF 全中招, 全部 version=2, 源头是直接写库的
scripts/process_pdfs.py 当时没传 hash)。

哈希从 UPLOAD_DIR 下该文档的 PDF 现算 —— 与上传口算的是同一个裸 SHA256。

用法:
  .venv/Scripts/python.exe scripts/backfill_content_hash.py          # 只报告
  .venv/Scripts/python.exe scripts/backfill_content_hash.py --apply  # 真写
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.database import db_manager
from src.infra.paths import DATA_REPORTS_DIR


def main() -> int:
    apply = "--apply" in sys.argv
    todo, missing = [], []
    for d in db_manager.get_all_documents():
        if d.get("content_hash"):
            continue
        path = os.path.join(DATA_REPORTS_DIR, d["filename"])
        if not os.path.exists(path):
            missing.append(d)
            continue
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        todo.append((d, h))

    for d, h in todo:
        print(f"  {'回填' if apply else '待回填'} id={d['id']} {d['filename']} → {h[:16]}")
        if apply:
            db_manager.update_content_hash(d["id"], h)
    for d in missing:
        print(f"  ⚠️  跳过 id={d['id']} {d['filename']}: 磁盘上找不到 PDF, 算不出哈希")

    print(f"\n{'已回填' if apply else '需要回填'} {len(todo)} 条, 跳过 {len(missing)} 条")
    if todo and not apply:
        print("加 --apply 真写")
    return 0


if __name__ == "__main__":
    sys.exit(main())
