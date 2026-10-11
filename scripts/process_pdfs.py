"""直接调用pipeline_manager处理PDF（绕过API认证）"""
import sys
sys.path.insert(0, r"D:\aiproject")

import hashlib
import os
from src.core.pipeline_manager import preview_document
from src.db.manager import db_manager

pdf_dir = r"D:\aiproject\data\kb_pending\source"

for f in sorted(os.listdir(pdf_dir)):
    if not f.endswith('.pdf'):
        continue
    path = os.path.join(pdf_dir, f)
    print(f"\n处理: {f}")

    # content_hash 必须传: 少了它, add_document 的"同内容跳过"和上传口的
    # find_document_by_hash 会**同时**失效, 同一份 PDF 再传一次就变成新版本,
    # 内容在 Milvus/BM25 里存两份。这 6 份 PDF 当初就是这么中的招。
    with open(path, "rb") as fh:
        content_hash = hashlib.sha256(fh.read()).hexdigest()

    # 先登记文档
    doc_id = db_manager.add_document(
        filename=f,
        status="pending_review",
        file_size=os.path.getsize(path),
        content_hash=content_hash,
        source_label="local_upload"
    )
    print(f"  登记文档 doc_id={doc_id}")

    # 预览处理（解析+分块，不写索引）
    result = preview_document(path, doc_id)
    if result["status"] == "success":
        print(f"  分块完成: {result['chunk_count']} 块, 状态=待审核")
    else:
        print(f"  失败: {result.get('error')}")

print("\n全部处理完成")
