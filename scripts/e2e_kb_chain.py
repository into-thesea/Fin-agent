"""
知识库链路端到端验收 —— 「上传 → 可检索 → 删除 → 重建不复活」

覆盖 2026-09-13「先修坏的」里的两个包:
  包 2  上传 ETL 必须把分块同时写进 Milvus (此前只写 FAISS → 检索永远搜不到新文档)
  包 3  删除必须清干净 分块文件 / BM25 / FAISS / Milvus (此前分块文件是追加型且从不清理
        → 下一次任何重建都会把已删文档重新写回索引，删除看起来生效、重建一次全复活)

前提:
  docker compose up -d            # redis + milvus 三件套 + neo4j 都要健康
  .venv/Scripts/python.exe run.py # API + Celery worker 都在跑

用法:
  .venv/Scripts/python.exe scripts/e2e_kb_chain.py
退出码: 0 = 全部通过, 1 = 有断言失败
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = os.getenv("E2E_API", "http://127.0.0.1:8001")
MARKER = "Xinghai Test Term 7788"      # 专有标记词, 保证 BM25 能唯一命中
PDF_NAME = "e2e_kb_chain_probe.pdf"

_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{detail}]" if detail else ""))
    return ok


# ── HTTP 小工具 (stdlib, 不引额外依赖) ──────────────────────

def _req(method: str, path: str, token: str = "", body: bytes = None,
         headers: dict = None) -> tuple:
    h = dict(headers or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + path, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:300].decode("utf-8", "replace")}


def make_pdf(path: str) -> None:
    """生成一份最小可解析的探针 PDF (纯 ASCII)。

    不依赖 PyMuPDF / reportlab —— 本环境两者都没装 (PyMuPDF 只是解析器里的代码回退,
    并未列进 requirements)。这里手写 PDF 结构并算出正确的 xref 偏移, pdfplumber
    能正常抽取文本。页文本必须 ≥50 字符, 否则解析器会静默跳过该页。
    """
    lines = [
        "Xinghai Internal Test Document",
        "",
        f"{MARKER} is a proprietary marker phrase used solely for the",
        "knowledge base end-to-end verification pipeline.",
        "It appears in exactly one uploaded document.",
        "",
        "Section Two: secondary content",
        "This paragraph exists only to give the chunker more than one section.",
        "Additional filler text keeps the extracted page length above the",
        "parser minimum so this page is not silently skipped.",
    ]
    content = "BT /F1 11 Tf 72 720 Td 16 TL\n"
    for ln in lines:
        esc = ln.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content += f"({esc}) Tj T*\n"
    content += "ET"
    stream = content.encode("latin-1")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()

    with open(path, "wb") as f:
        f.write(bytes(out))


# ── 各存储的直接探针 ────────────────────────────────────────

def milvus_count_for(filename: str) -> int:
    """Milvus 里 document_id == filename 的条数"""
    from src.vectorstore.milvus_manager import MilvusManager
    mgr = MilvusManager()
    if not mgr.available:
        raise RuntimeError("Milvus 不可用 —— 前提条件未满足")
    col = mgr._collection
    res = col.query(expr=f'document_id == "{filename}"', output_fields=["id"])
    return len(res)


def jsonl_count_for(filename: str) -> int:
    from src.infra.paths import CHUNKS_PROCESSED_PATH
    if not os.path.exists(CHUNKS_PROCESSED_PATH):
        return 0
    n = 0
    with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("source") == filename:
                    n += 1
            except json.JSONDecodeError:
                continue
    return n


def bm25_hits_for(query: str, filename: str, top_k: int = 20) -> int:
    """BM25 检索该 query, 统计命中某文档的分块数"""
    from src.infra.paths import BM25_INDEX_PATH
    from src.retrieval.bm25_index import BM25Index
    idx = BM25Index()
    idx.load(BM25_INDEX_PATH)
    return sum(1 for r in idx.search(query, top_k=top_k)
               if r.get("source") == filename or r.get("chunk_id", "").startswith(filename))


# ── 主流程 ─────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("知识库链路端到端验收")
    print("=" * 60)

    # 0) 前置: API 可达
    print("\n[0] 前置检查")
    try:
        st, _ = _req("GET", "/health")
    except Exception as e:
        print(f"  ❌ API 不可达 ({API}): {e}\n  请先运行 run.py")
        return 1
    if not check("API 可达", st == 200, f"status={st}"):
        return 1

    from src.config import settings
    if not settings.admin_password:
        print("  ❌ .env 未配置 ADMIN_PASSWORD, 无法登录")
        return 1
    st, body = _req("POST", "/api/v1/auth/login",
                    body=json.dumps({"username": settings.admin_username,
                                     "password": settings.admin_password}).encode(),
                    headers={"Content-Type": "application/json"})
    token = body.get("token", "")
    if not check("管理员登录", st == 200 and bool(token), f"status={st}"):
        return 1

    # 1) 准备探针 PDF
    print("\n[1] 准备探针文档")
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", PDF_NAME)
    pdf_path = os.path.abspath(pdf_path)
    make_pdf(pdf_path)
    check("生成探针 PDF", os.path.exists(pdf_path), os.path.basename(pdf_path))

    # 先清一次: 上一轮跑崩可能留下残留
    st, docs = _req("GET", "/api/v1/knowledge/documents", token)
    for d in docs.get("documents", []):
        if d.get("filename") == PDF_NAME:
            _req("DELETE", f"/api/v1/knowledge/documents/{d['id']}", token)
            print("    (清理上一轮残留)")

    # 2) 上传
    print("\n[2] 上传")
    with open(pdf_path, "rb") as f:
        content = f.read()
    boundary = "----e2ekbchain"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{PDF_NAME}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    st, res = _req("POST", "/api/v1/knowledge/batch-upload", token, body,
                   {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    r0 = (res.get("results") or [{}])[0]
    if not check("上传被接受", st == 200 and r0.get("status") == "accepted",
                 f"status={st} result={r0.get('status')} msg={r0.get('message', '')}"):
        print("    ⚠️  status=rejected 通常意味着 Celery Worker 没在跑（这是刻意的：不再降级同步线程）")
        return 1
    task_id = r0["task_id"]

    # 3) 等 ETL 完成
    print("\n[3] 等 ETL 完成")
    deadline = time.time() + 300
    final = None
    while time.time() < deadline:
        st, t = _req("GET", f"/api/v1/tasks/{task_id}", token)
        final = t
        if t.get("status") in ("completed", "failed"):
            break
        time.sleep(3)
    if not check("ETL 完成", final and final.get("status") == "completed",
                 f"status={final and final.get('status')} err={final and final.get('error')}"):
        return 1

    # 4) 上传后的可见性 —— 这是包 2 的核心验收
    print("\n[4] 上传后: 分块必须进了每一个检索通路")
    check("Milvus 有该文档的向量", milvus_count_for(PDF_NAME) > 0,
          f"{milvus_count_for(PDF_NAME)} 条")
    check("分块文件有该文档", jsonl_count_for(PDF_NAME) > 0,
          f"{jsonl_count_for(PDF_NAME)} 条")
    check("BM25 能检索到该文档", bm25_hits_for(MARKER, PDF_NAME) > 0)
    st, docs = _req("GET", "/api/v1/knowledge/documents", token)
    doc = next((d for d in docs.get("documents", []) if d.get("filename") == PDF_NAME), None)
    if not check("文档列表可见", doc is not None):
        return 1

    # 5) 端到端检索
    print("\n[5] 端到端检索")
    from src.retrieval.retriever import HybridRetriever
    ctx = HybridRetriever().hybrid_retrieve(f"{MARKER} 是什么", top_k=5)
    sources = [c.get("source", "") for c in ctx.get("local", [])]
    check("混合检索命中探针文档", PDF_NAME in sources, f"top5 sources={sources[:3]}")

    # 6) 删除
    print("\n[6] 删除")
    st, res = _req("DELETE", f"/api/v1/knowledge/documents/{doc['id']}", token)
    if not check("删除接口返回成功", st == 200 and res.get("deleted") is True,
                 f"status={st} resp={str(res)[:160]}"):
        return 1

    # 7) 删除后: 每一处都必须干净
    print("\n[7] 删除后: 四处存储都必须清干净")
    check("Milvus 已无该文档向量", milvus_count_for(PDF_NAME) == 0,
          f"{milvus_count_for(PDF_NAME)} 条")
    check("分块文件已无该文档", jsonl_count_for(PDF_NAME) == 0,
          f"{jsonl_count_for(PDF_NAME)} 条")
    check("BM25 已无该文档", bm25_hits_for(MARKER, PDF_NAME) == 0)
    st, docs = _req("GET", "/api/v1/knowledge/documents", token)
    check("文档列表已移除",
          not any(d.get("filename") == PDF_NAME for d in docs.get("documents", [])))

    # 8) 复活测试 —— 包 3 的核心验收
    print("\n[8] 复活测试: 手动重建 BM25 索引 (模拟后续任何一次重建)")
    from src.infra.paths import CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH
    from src.retrieval.bm25_index import rebuild_bm25_index
    rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)
    check("重建后 BM25 仍无该文档 (未复活)", bm25_hits_for(MARKER, PDF_NAME) == 0)
    check("重建后分块文件仍无该文档", jsonl_count_for(PDF_NAME) == 0)

    # 收尾
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    print("\n" + "=" * 60)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"结果: {len(_results) - len(failed)}/{len(_results)} 通过")
    if failed:
        print("失败项: " + ", ".join(failed))
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
