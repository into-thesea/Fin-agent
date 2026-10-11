import sys
sys.path.insert(0, r"D:\aiproject")
import os
from src.core.pipeline_manager import preview_document, PENDING_DIR

pdf_dir = r"D:\aiproject\data\kb_pending\source"
pdf_map = {
    566: "01_江苏银行_聚宝财富理财产品说明书.pdf",
    567: "02_建设银行_兴银理财增盈稳享产品说明书.pdf",
    568: "03_建设银行_理财产品风险揭示书和说明书.pdf",
    569: "04_中银理财_理财产品文件.pdf",
    570: "05_农业银行_理财产品说明书.pdf",
    571: "06_建信理财_理财产品文件.pdf",
}

for doc_id, fname in pdf_map.items():
    path = os.path.join(pdf_dir, fname)
    # 删旧文件
    old = os.path.join(PENDING_DIR, f"{doc_id}_chunks.jsonl")
    if os.path.exists(old):
        os.remove(old)
    # 重新生成
    result = preview_document(path, doc_id)
    if result["status"] == "success":
        n_parent = sum(1 for c in result["chunks"] if c.get("chunk_type") == "parent")
        n_child = sum(1 for c in result["chunks"] if c.get("chunk_type") == "child")
        print(f"doc_id={doc_id} {fname[:30]:30s}: 总{result['chunk_count']} = 父{n_parent} + 子{n_child}")
    else:
        print(f"doc_id={doc_id} {fname[:30]:30s}: 失败 - {result.get('error')}")
