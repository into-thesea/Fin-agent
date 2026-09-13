"""知识库管理台端到端验收

覆盖三件事 (按 spec 第八章):
  1. 检索参数保存后**免重启**立即生效
  2. 切分参数只影响**之后新上传**的文档
  3. 索引维护重建后向量库与分块数仍一致

前提:
  docker compose up -d             # redis + milvus + neo4j 都要健康
  .venv/Scripts/python.exe run.py  # API + Celery Worker 都在跑

用法:
  .venv/Scripts/python.exe scripts/e2e_kb_console.py
退出码: 0 = 全部通过, 1 = 有断言失败

脚本会 PUT data/kb_settings.json, 所以**结束时(含中途失败)一律恢复代码默认值**,
并删掉上传的探针文档与本地临时 PDF —— 连跑两次结果必须相同。
"""

import json
import os
import sys
import time
from urllib.parse import quote

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from e2e_kb_chain import _req, API                        # noqa: E402
from src.config import settings                           # noqa: E402

PDF_A = "e2e_kb_console_default.pdf"   # 改参数**之前**上传的对照组文档
PDF_B = "e2e_kb_console_custom.pdf"    # 改参数**之后**上传的实验组文档
QUERY = "稳盈添利30天风险等级是多少"
CUSTOM_CHUNKING = {"chunk_size": 200, "overlap": 40, "max_chunk_content": 150}
JSON_H = {"Content-Type": "application/json"}

_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{detail}]" if detail else ""))
    return ok


# ── 探针 PDF (stdlib, 不引额外依赖) ────────────────────────

def probe_lines(tag: str, n: int = 8) -> list:
    """探针正文: 单段连续散文, 无标题行。

    每行 ~110 字符、正文 ~880 字符, 刻意同时超过默认 chunk_size(350) 与本次要改成的
    200 —— 这样"切分参数换了"必然在**块数**与**最长块**上留下可比较的差异。
    """
    return [
        f"Probe {tag} line {i:02d}: "
        "the knowledge base console acceptance run uses this filler sentence so the "
        "extracted page text grows past the chunk size under test and forces a split."
        for i in range(n)
    ]


def make_pdf(path: str, lines: list) -> None:
    """生成一份最小可解析的探针 PDF (纯 ASCII)。

    与 e2e_kb_chain.make_pdf 同法: 手写 PDF 结构并算好 xref 偏移, 不依赖任何 PDF 库。
    字号刻意压到 9pt —— pdfplumber 按页面框裁剪, 超宽的长行会被整段丢掉。
    """
    content = "BT /F1 9 Tf 72 720 Td 14 TL\n"
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


# ── 接口小封装 ─────────────────────────────────────────────

def put_section(token: str, section: str, values: dict) -> tuple:
    return _req("PUT", "/api/v1/kb/strategy", token,
                json.dumps({"section": section, "values": values}).encode(), JSON_H)


def upload(token: str, path: str) -> tuple:
    """返回 (http_status, batch-upload 里该文件的结果)"""
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    boundary = "----e2ekbconsole"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    st, res = _req("POST", "/api/v1/knowledge/batch-upload", token, body,
                   {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return st, (res.get("results") or [{}])[0]


def wait_task(token: str, task_id: str, timeout: int, every: int = 3) -> dict:
    deadline = time.time() + timeout
    final: dict = {}
    while time.time() < deadline:
        _, final = _req("GET", f"/api/v1/tasks/{task_id}", token=token)
        if final.get("status") in ("completed", "failed"):
            break
        time.sleep(every)
    return final


def doc_chunk_stats(token: str, filename: str) -> tuple:
    """某文档在分块文件里的 (条数, 最长块字符数) —— 走 /kb/chunks 的 source 过滤"""
    _, r = _req("GET", f"/api/v1/kb/chunks?source={quote(filename)}&size=100", token=token)
    items = r.get("items") or []
    longest = max([i.get("length") or 0 for i in items], default=0)
    return r.get("total", 0), longest


def delete_by_name(token: str, filename: str) -> dict:
    """按文件名找到文档并删除 (不关心有没有, 幂等)"""
    _, docs = _req("GET", "/api/v1/knowledge/documents", token=token)
    doc = next((d for d in docs.get("documents", []) if d.get("filename") == filename), None)
    if not doc:
        return {"deleted": True, "message": "不存在, 无需删除"}
    st, res = _req("DELETE", f"/api/v1/knowledge/documents/{doc['id']}", token)
    if st != 200:
        return {"deleted": False, "message": f"status={st} resp={str(res)[:160]}"}
    return res


# ── 三段验收 ───────────────────────────────────────────────

def check_retrieval_hot_reload(token: str, before: dict) -> None:
    """1. 检索参数保存后免重启立即生效"""
    print("\n[1] 检索参数免重启生效")
    _, orig = _req("POST", "/api/v1/kb/retrieval-test", token,
                   json.dumps({"query": QUERY, "top_k": 5}).encode(), JSON_H)
    n_before = len(orig.get("fused") or [])
    rrf_before = (orig.get("config_used") or {}).get("rrf_k")
    score_before = (orig.get("fused") or [{}])[0].get("rrf_score")

    # rrf_k 也一起改: top_k 只影响本次请求的截断, 证明不了"配置被读到" ——
    # 而 rrf_k 是 retrieve_debug 现读 data/kb_settings.json 拿到的, 且真的进融合计算。
    values = {**before["retrieval"]["current"], "top_k": 3, "rrf_k": 1}
    st, saved = put_section(token, "retrieval", values)
    if not check("保存检索参数", st == 200, f"status={st} resp={str(saved)[:160]}"):
        return
    check("标注为下次请求生效(免重启)", saved.get("applies_to") == "next_request",
          f"applies_to={saved.get('applies_to')}")

    _, after = _req("POST", "/api/v1/kb/retrieval-test", token,
                    json.dumps({"query": QUERY, "top_k": 3}).encode(), JSON_H)
    n_after = len(after.get("fused") or [])
    check("同一次运行内新参数即生效", n_after <= 3 and n_before > 3,
          f"融合条数 {n_before} → {n_after} (请求 top_k 5 → 3)")

    rrf_after = (after.get("config_used") or {}).get("rrf_k")
    check("服务进程未重启就读到新配置", rrf_after == 1,
          f"config_used.rrf_k {rrf_before} → {rrf_after} (期望 1)")

    score_after = (after.get("fused") or [{}])[0].get("rrf_score")
    check("新 rrf_k 真的进了融合计算", score_after != score_before,
          f"首位 rrf_score {score_before} → {score_after}")

    # 立刻恢复, 免得影响后面的检索
    put_section(token, "retrieval", before["retrieval"]["default"])


def check_chunking_scope(token: str, before: dict) -> tuple:
    """2. 切分参数只影响之后新上传的文档。返回 (对照组块数, 最长块)"""
    print("\n[2] 切分参数只影响之后新上传的文档")

    # 对照组: 用**当前(默认)**参数先传一份, 作为"旧文档"被比较
    path_a = os.path.join(REPO_ROOT, PDF_A)
    make_pdf(path_a, probe_lines("default"))
    st, r = upload(token, path_a)
    if not check("对照组文档上传被接受", st == 200 and r.get("status") == "accepted",
                 f"status={st} result={r.get('status')} msg={r.get('message', '')}"):
        return 0, 0
    final = wait_task(token, r["task_id"], timeout=300)
    if not check("对照组 ETL 完成", final.get("status") == "completed",
                 f"status={final.get('status')} err={final.get('error')}"):
        return 0, 0
    n_a, max_a = doc_chunk_stats(token, PDF_A)

    # 换切分参数
    st, s2 = put_section(token, "chunking",
                         {**before["chunking"]["current"], **CUSTOM_CHUNKING})
    if not check("保存切分参数", st == 200, f"status={st} resp={str(s2)[:160]}"):
        return n_a, max_a
    check("标注为下次上传生效(不回溯)", s2.get("applies_to") == "next_upload",
          f"applies_to={s2.get('applies_to')}")

    # 实验组: 内容必须与对照组不同 —— 同内容会被整文件哈希去重直接跳过
    path_b = os.path.join(REPO_ROOT, PDF_B)
    make_pdf(path_b, probe_lines("custom"))
    st, r = upload(token, path_b)
    if not check("实验组文档上传被接受", st == 200 and r.get("status") == "accepted",
                 f"status={st} result={r.get('status')} msg={r.get('message', '')}"):
        return n_a, max_a
    final = wait_task(token, r["task_id"], timeout=300)
    if not check("实验组 ETL 完成", final.get("status") == "completed",
                 f"status={final.get('status')} err={final.get('error')}"):
        return n_a, max_a
    n_b, max_b = doc_chunk_stats(token, PDF_B)

    check("新上传按新 chunk_size 切分(块数变多)", n_b > n_a and n_b > 0,
          f"默认参数 {n_a} 块 → 新参数 {n_b} 块 (chunk_size 350 → 200)")
    check("新上传的单块长度受新参数约束", 0 < max_b <= CUSTOM_CHUNKING["chunk_size"],
          f"最长块 {max_b} 字符 (chunk_size={CUSTOM_CHUNKING['chunk_size']})")
    check("旧文档仍按旧参数, 没被回溯重切", max_a > CUSTOM_CHUNKING["chunk_size"],
          f"对照组最长块 {max_a} 字符 (期望 >200, 即仍是旧的 350 上限)")

    # 恢复切分参数 (再往后就只剩重建与收尾, 不该带着非默认配置跑)
    put_section(token, "chunking", before["chunking"]["default"])
    return n_a, max_a


def check_rebuild(token: str) -> None:
    """3. 索引维护重建"""
    print("\n[3] 索引维护重建")
    st, r = _req("POST", "/api/v1/kb/rebuild", token)
    if not check("重建任务已受理", st == 200 and r.get("task_id"),
                 f"status={st} resp={str(r)[:160]}"):
        return
    tid = r["task_id"]
    # 全量重建要重新编码 200+ 个分块, 实测 2~4 分钟, 轮询给足 10 分钟
    final = wait_task(token, tid, timeout=600, every=5)
    if not check("重建完成", final.get("status") == "completed",
                 f"status={final.get('status')} err={final.get('error')}"):
        return

    _, chunks = _req("GET", "/api/v1/kb/chunks?page=1&size=1", token=token)
    total = chunks.get("total", 0)
    check("重建后分块数 > 0", total > 0, f"{total} 条")

    from src.vectorstore.milvus_manager import MilvusManager
    # 必须在重建**之后**才实例化: 它是进程内单例, 重建会把 collection drop 重建,
    # 提前拿到的句柄会指向已被删掉的旧 collection, 读出来的数是错的
    mgr = MilvusManager()
    if not check("Milvus 可用", mgr.available, f"{mgr.available}"):
        return
    check("重建后向量库与分块数一致", mgr.total_count == total,
          f"milvus={mgr.total_count} chunks={total}")


# ── 收尾 ───────────────────────────────────────────────────

def cleanup(token: str, before: dict) -> None:
    """无论成功失败都跑: 删探针文档 + 恢复默认参数 (脚本可重复运行的前提)"""
    print("\n[4] 收尾")
    for name in (PDF_A, PDF_B):
        res = delete_by_name(token, name)
        check(f"删除探针文档 {name}", res.get("deleted") is True, str(res.get("message", ""))[:120])
        n, _ = doc_chunk_stats(token, name)
        check(f"其分块已清干净 {name}", n == 0, f"残留 {n} 条")

    for section in ("retrieval", "chunking"):
        if section in before:
            put_section(token, section, before[section]["default"])
    _, restored = _req("GET", "/api/v1/kb/strategy", token=token)
    for section in ("retrieval", "chunking"):
        changed = (restored.get(section) or {}).get("changed")
        check(f"{section} 已恢复默认(无偏离项)", changed == [], f"偏离 {changed}")

    for name in (PDF_A, PDF_B):
        p = os.path.join(REPO_ROOT, name)
        if os.path.exists(p):
            os.remove(p)
    check("本地临时 PDF 已删除",
          not any(os.path.exists(os.path.join(REPO_ROOT, n)) for n in (PDF_A, PDF_B)))


def main() -> int:
    print("=" * 60)
    print("知识库管理台 端到端验收")
    print("=" * 60)

    print("\n[0] 前置检查")
    try:
        st, _ = _req("GET", "/health")
    except Exception as e:
        print(f"  ❌ API 不可达 ({API}): {e}\n  请先运行 .venv/Scripts/python.exe run.py")
        return 1
    if not check("API 可达", st == 200, f"status={st}"):
        return 1

    if not settings.admin_password:
        print("  ❌ .env 未配置 ADMIN_PASSWORD, 无法登录")
        return 1
    st, body = _req("POST", "/api/v1/auth/login",
                    body=json.dumps({"username": settings.admin_username,
                                     "password": settings.admin_password}).encode(),
                    headers=JSON_H)
    token = body.get("token", "")
    if not check("管理员登录", st == 200 and bool(token), f"status={st}"):
        return 1

    st, sysinfo = _req("GET", "/api/v1/system/status", token=token)
    services = (sysinfo or {}).get("services", {})
    # 上传与重建都硬要求 Worker 在跑(已不再降级到同步线程), 缺了后面全是假失败
    if not check("Celery Worker 在跑", services.get("celery_worker") == "ok", str(services)):
        print("  ⚠️  上传与重建都要求 Worker: .venv/Scripts/python.exe run.py")
        return 1

    _, before = _req("GET", "/api/v1/kb/strategy", token=token)
    if not check("读取策略配置", "retrieval" in before and "chunking" in before,
                 f"status={st} keys={sorted(before)[:4]}"):
        return 1

    try:
        check_retrieval_hot_reload(token, before)
        check_chunking_scope(token, before)
        check_rebuild(token)
    finally:
        try:
            cleanup(token, before)
        except Exception as e:                       # 收尾崩了也要给出退出码, 不能吞
            print(f"  ❌ 收尾异常: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"结果: {len(_results) - len(failed)}/{len(_results)} 通过")
    if failed:
        print("失败项: " + ", ".join(failed))
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
