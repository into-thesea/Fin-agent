"""临时校验: 重复内容只能入库一次(幂等)。

判据是决定性的: 先单份入库量出「一份是多少」作为基准, 清空后用**同批次两份完全
相同的文件**再传一次, 最终计数必须与基准一致, 且文档表只有一行。
(只看「Milvus 条数 == 分块条数」是不够的 —— 两边一起翻倍也会通过。)

用法: .venv/Scripts/python.exe scripts/_check_idempotency.py
(需要 API + Celery Worker 在跑)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from e2e_kb_chain import _req, make_pdf, PDF_NAME          # noqa: E402
from src.config import settings                             # noqa: E402
from src.infra.paths import CHUNKS_PROCESSED_PATH           # noqa: E402
from src.vectorstore.milvus_manager import MilvusManager    # noqa: E402


def login() -> str:
    _, body = _req("POST", "/api/v1/auth/login",
                   body=json.dumps({"username": settings.admin_username,
                                    "password": settings.admin_password}).encode(),
                   headers={"Content-Type": "application/json"})
    return body["token"]


def purge(token: str) -> None:
    _, docs = _req("GET", "/api/v1/knowledge/documents", token=token)
    for d in docs.get("documents", []):
        if d.get("filename") == PDF_NAME:
            _req("DELETE", f"/api/v1/knowledge/documents/{d['id']}", token)


def snapshot() -> tuple:
    mgr = MilvusManager()
    assert mgr.available, "Milvus 不可用"
    n_milvus = len(mgr._collection.query(expr=f'document_id == "{PDF_NAME}"',
                                         output_fields=["id"]))
    n_jsonl = sum(1 for line in open(CHUNKS_PROCESSED_PATH, encoding="utf-8")
                  if line.strip() and json.loads(line).get("source") == PDF_NAME)
    return n_milvus, n_jsonl


def upload(token: str, copies: int) -> dict:
    p = os.path.abspath(PDF_NAME)
    content = open(p, "rb").read()
    b = "----idem"
    parts = b""
    for _ in range(copies):
        parts += (
            f'--{b}\r\nContent-Disposition: form-data; name="files"; '
            f'filename="{PDF_NAME}"\r\nContent-Type: application/pdf\r\n\r\n'
        ).encode() + content + b"\r\n"
    parts += f"--{b}--\r\n".encode()
    st, res = _req("POST", "/api/v1/knowledge/batch-upload", token, parts,
                   {"Content-Type": f"multipart/form-data; boundary={b}"})
    for r in res.get("results", []):
        if r.get("task_id"):
            for _ in range(60):
                _, t = _req("GET", f"/api/v1/tasks/{r['task_id']}", token=token)
                if t.get("status") in ("completed", "failed"):
                    break
                time.sleep(3)
    return res


def doc_rows(token: str) -> int:
    _, docs = _req("GET", "/api/v1/knowledge/documents", token=token)
    return sum(1 for d in docs.get("documents", []) if d.get("filename") == PDF_NAME)


def main() -> int:
    token = login()
    p = os.path.abspath(PDF_NAME)
    make_pdf(p)

    # ── 基准: 单份入库 ──
    purge(token)
    upload(token, 1)
    base_milvus, base_jsonl = snapshot()
    print(f"基准(单份): Milvus={base_milvus} 分块文件={base_jsonl} 文档行={doc_rows(token)}")
    if base_milvus == 0:
        print("❌ 基准为 0, 环境有问题")
        return 1

    # ── 同批次两份完全相同的文件 ──
    purge(token)
    res = upload(token, 2)
    n_milvus, n_jsonl = snapshot()
    n_docs = doc_rows(token)
    print(f"重复批次: API 返回 accepted={res.get('accepted')} skipped={res.get('skipped')}")
    print(f"重复批次后: Milvus={n_milvus} 分块文件={n_jsonl} 文档行={n_docs}")

    ok = (n_docs == 1 and n_milvus == base_milvus and n_jsonl == base_jsonl)
    print("结论:", "✅ 幂等生效（重复内容只入库一次）" if ok
          else "❌ 重复入库（计数相对基准翻倍）")

    purge(token)
    if os.path.exists(p):
        os.remove(p)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
