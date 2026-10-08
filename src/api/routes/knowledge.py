"""
Fin-Agent 知识库路由 (文档管理 + ETL 任务追踪)

提供:
  - POST /api/v1/knowledge/upload             异步上传 PDF（单文件）
  - POST /api/v1/knowledge/batch-upload       异步批量上传 PDF（多文件）
  - DELETE /api/v1/knowledge/documents/{id}   删除文档（含 SQLite/Milvus/PDF/Celery）
  - GET  /api/v1/knowledge/documents          文档列表
  - GET  /api/v1/knowledge/stats              知识库统计
  - GET  /api/v1/tasks/{task_id}              查询异步任务状态
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.api.routes.auth import require_roles

logger = logging.getLogger(__name__)

# 后端强制鉴权: 知识库管理仅管理员可操作
router = APIRouter(prefix="/api/v1/knowledge", tags=["知识库"],
                   dependencies=[Depends(require_roles("admin"))])

# 同步 I/O 辅助: 在默认线程池中执行, 不阻塞事件循环
async def _aio(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

# 上传配置
from src.infra.paths import DATA_REPORTS_DIR

UPLOAD_DIR = DATA_REPORTS_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
READ_CHUNK = 1024 * 1024  # 1MB
MAX_BATCH_FILES = 20        # 单次批量上传的文件数上限


# ──────────────────────────────────────────────
# 单文件上传
# ──────────────────────────────────────────────


async def _process_upload(file: UploadFile) -> dict:
    """处理单个文件上传：校验 → 去重 → 落盘 → 提交 ETL

    Celery Worker 是硬要求, 不再降级到后台线程 —— 两条路原先跑的是两套分块
    (按页切 vs 章节感知) 和两套索引流程, 同一份 PDF 因「当时 worker 在不在」
    而产出不同结果。环境没准备好就该直接报错, 而不是给一个更差的答案。
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        return {"filename": file.filename or "unknown", "status": "rejected",
                "message": "仅支持 PDF 文件"}

    # 边读边算哈希并校验大小: 原先 `await file.read()` 先把整个文件读进内存,
    # 再判断 50MB 上限 —— N 个文件并发就是 N×文件大小 的常驻内存。
    digest = hashlib.sha256()
    total = 0
    parts: list[bytes] = []
    while True:
        part = await file.read(READ_CHUNK)
        if not part:
            break
        total += len(part)
        if total > MAX_FILE_SIZE:
            return {"filename": file.filename, "status": "rejected",
                    "message": "文件大小超过 50MB 限制"}
        digest.update(part)
        parts.append(part)

    content_hash = digest.hexdigest()  # 与 db_manager.compute_file_hash 同为裸 SHA256
    safe_name = file.filename.replace(" ", "_").replace("/", "_")

    from src.database import db_manager
    existing = db_manager.find_document_by_hash(content_hash)
    if existing:
        # 关键: 这里不能「先落盘、再去重、重复就删」。
        # 同名同内容重复上传时, 盘上那个文件正是**已入库文档的 PDF 源文件**,
        # 删掉它会让已入库文档的源文件消失, 之后删除该文档也找不到 PDF。
        # 去重通过前不碰磁盘即可根除。
        logger.info("文件重复 (hash=%s): %s", content_hash[:12], safe_name)
        return {
            "filename": safe_name,
            "status": "skipped",
            "message": "内容已存在，跳过入库",
            "doc_id": existing["id"],
            "content_hash": content_hash,
        }

    from src.celery_app import is_celery_worker_running
    if not is_celery_worker_running():
        logger.error("拒绝上传: ETL Worker 未运行 (file=%s)", safe_name)
        return {
            "filename": safe_name,
            "status": "rejected",
            "message": "ETL Worker 未运行，无法处理上传。请启动 Celery Worker 后重试。",
        }

    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as f:
        for part in parts:
            f.write(part)

    version_tag = f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    from src.tasks.etl_tasks import etl_pipeline_task
    task = etl_pipeline_task.delay(
        pdf_path=file_path.replace("\\", "/"),
        filename=safe_name,
        version_tag=version_tag,
    )

    # 记录 task_id → task_logs, 删除文档时的 Celery revoke 依赖它。
    # 原先这张表无人写入, 导致「取消正在跑的任务」是死代码。
    try:
        db_manager.create_task(task.id, safe_name)
    except Exception as e:
        logger.error("task_logs 写入失败 (%s): %s", task.id, e)

    logger.info(
        "上传任务已提交: file=%s, task_id=%s",
        safe_name, task.id,
        extra={"task_id": task.id, "file": safe_name},
    )
    return {
        "filename": safe_name,
        "status": "accepted",
        "task_id": task.id,
        "content_hash": content_hash,
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """异步上传单文件 PDF"""
    result = await _process_upload(file)
    if result.get("status") == "rejected":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ──────────────────────────────────────────────
# 批量上传
# ──────────────────────────────────────────────


@router.post("/batch-upload")
async def batch_upload(files: list[UploadFile] = File(...)):
    """
    异步批量上传 PDF 文件

    接收多个文件，逐个校验 → 去重 → 提交 ETL 任务。
    所有文件并行处理，返回每个文件的结果列表。

    Returns:
        {"total": int, "results": [{filename, status, task_id?, doc_id?, message?}]}
    """
    if not files:
        raise HTTPException(status_code=400, detail="未选择任何文件")
    # 并发闸: 原先无任何上限, 前端一次全塞、后端 asyncio.gather 全并行。
    # 超出就明确拒绝, 而不是把 N 个文件的解析与向量化一次性压进内存和 Worker 队列。
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多上传 {MAX_BATCH_FILES} 个文件（当前 {len(files)} 个），请分批上传",
        )

    import asyncio
    _raw_results = await asyncio.gather(
        *[_process_upload(f) for f in files],
        return_exceptions=True,
    )
    results = []
    for i, r in enumerate(_raw_results):
        if isinstance(r, Exception):
            fname = files[i].filename or f"file_{i}"
            logger.error("文件处理异常 (%s): %s", fname, r)
            results.append({
                "filename": fname,
                "status": "rejected",
                "message": f"服务器内部错误: {str(r)[:200]}",
            })
        else:
            results.append(r)
    accepted = sum(1 for r in results if r["status"] == "accepted")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    rejected = sum(1 for r in results if r["status"] == "rejected")

    logger.info("批量上传完成: total=%d, accepted=%d, skipped=%d, rejected=%d",
                len(results), accepted, skipped, rejected)

    return {
        "total": len(results),
        "accepted": accepted,
        "skipped": skipped,
        "rejected": rejected,
        "results": results,
    }


# ──────────────────────────────────────────────
# 文档列表 & 统计
# ──────────────────────────────────────────────


@router.get("/documents")
async def list_documents():
    """文档列表"""
    from src.database import db_manager

    docs = await _aio(db_manager.get_all_documents)
    return {"total": len(docs), "documents": docs}


@router.get("/stats")
async def knowledge_stats():
    """知识库统计"""
    from src.database import db_manager

    return await _aio(db_manager.get_stats)


# ──────────────────────────────────────────────
# 文档删除
# ──────────────────────────────────────────────


def _remove_chunks_of(filename: str) -> int:
    """从全量分块文件里移除某文档的所有分块, 返回移除条数。

    这是原先完全缺失的一步: chunks_processed.jsonl 是**追加型**文件, 删文档时
    不清它, 下一次任何重建(BM25 / 向量索引)都会把已删
    文档的分块重新写回索引 —— 删除看起来生效了, 重建一次全复活。
    """
    import json

    from src.infra.paths import CHUNKS_PROCESSED_PATH
    if not os.path.exists(CHUNKS_PROCESSED_PATH):
        return 0
    kept, removed = [], 0
    with open(CHUNKS_PROCESSED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue  # 损坏行丢弃, 与 read_lines 行为一致
            if c.get("source") == filename:
                removed += 1
            else:
                kept.append(line)
    if removed:
        tmp = CHUNKS_PROCESSED_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CHUNKS_PROCESSED_PATH)
    return removed


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """
    删除文档及其所有关联数据

    清理顺序(有依赖): 分块文件 → BM25 重建 → Milvus 向量 → PDF → Celery → SQLite

    任一步失败则**不删 SQLite 记录**, 文档仍留在列表里可以重试 ——
    否则记录没了、索引里的残留却还在, 用户既看不到也删不掉。
    索引清理干净后才摘掉记录。

    Returns: {"deleted": bool, "message": str, "details": dict, "errors": dict}
    """
    from src.database import db_manager

    # 除 HTTPException 外的所有 I/O 操作在线程中执行
    def _delete_sync():
        doc = db_manager.get_document_by_id(doc_id)
        if not doc:
            return {"_404": True}

        filename = doc.get("filename", "")
        details = {"chunks": 0, "bm25": False, "milvus": 0,
                   "pdf": False, "celery": False, "sqlite": False}
        errors: dict = {}

        # 1. 分块文件 (必须在 BM25 重建之前)
        try:
            details["chunks"] = _remove_chunks_of(filename)
        except Exception as e:
            errors["chunks"] = str(e)
            logger.error("分块文件清理失败 (%s): %s", filename, e, exc_info=True)

        # 2. BM25 重建 —— 索引里不再有该文档
        try:
            from src.infra.paths import BM25_INDEX_PATH, CHUNKS_PROCESSED_PATH
            from src.retrieval.bm25_index import rebuild_bm25_index
            rebuild_bm25_index(CHUNKS_PROCESSED_PATH, BM25_INDEX_PATH)
            details["bm25"] = True
        except Exception as e:
            errors["bm25"] = str(e)
            logger.error("BM25 重建失败 (%s): %s", filename, e, exc_info=True)

        # 3. 向量: Milvus (唯一向量后端)
        #    key 必须是 filename —— 索引 metadata 里的 document_id 存的就是文件名
        #    (vector_indexer / pipeline_manager 都传 file_name)。
        #    原先传的是 SQLite 自增 id, 永远匹配不上, 实际删除 0 条。
        try:
            from src.vectorstore.milvus_manager import MilvusManager
            details["milvus"] = MilvusManager().delete_by_document(filename)
        except Exception as e:
            errors["milvus"] = str(e)
            logger.error("Milvus 删除失败 (%s): %s", filename, e, exc_info=True)

        # 4. PDF 源文件
        pdf_path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                details["pdf"] = True
            except OSError as e:
                errors["pdf"] = str(e)
                logger.error("PDF 删除失败 %s: %s", pdf_path, e)

        # 5. Celery 任务撤销 (task_id 由上传时写入 task_logs)
        try:
            from src.celery_app import celery_app
            cursor = db_manager.conn.cursor()
            cursor.execute(
                "SELECT task_id FROM task_logs WHERE filename = ? "
                "AND status NOT IN ('completed', 'failed')",
                (filename,),
            )
            task_row = cursor.fetchone()
            if task_row and task_row["task_id"]:
                celery_app.control.revoke(task_row["task_id"], terminate=True, signal="SIGTERM")
                details["celery"] = True
        except Exception as e:
            errors["celery"] = str(e)
            logger.error("Celery 撤销失败 (%s): %s", filename, e, exc_info=True)

        # 6. SQLite 记录 —— 只在索引侧全部清理成功后才摘掉
        if errors:
            logger.error("文档 %s 删除未完成, 保留记录以便重试: %s", filename, errors)
        else:
            details["sqlite"] = db_manager.delete_document(doc_id)

        return {"_404": False, "details": details, "errors": errors, "filename": filename}

    result = await _aio(_delete_sync)
    if result.get("_404"):
        raise HTTPException(status_code=404, detail=f"文档 ID {doc_id} 不存在")

    details = result["details"]
    errors = result["errors"]
    filename = result["filename"]

    if errors:
        raise HTTPException(
            status_code=500,
            detail=f"文档 '{filename}' 删除未完成, 已保留记录可重试。失败项: {errors}",
        )

    return {
        "deleted": details["sqlite"],
        "message": f"文档 '{filename}' 删除成功" if details["sqlite"] else "文档记录删除失败",
        "details": details,
    }


# ──────────────────────────────────────────────
# 获取已有文件列表
# ──────────────────────────────────────────────


@router.get("/existing-files")
async def list_existing_files():
    """
    列出 data_reports 目录中尚未入库的 PDF 文件

    Returns:
        {"files": [{"name": str, "size": int, "path": str, "in_db": bool}], ...}
    """
    import glob

    from src.database import db_manager

    pdf_files = glob.glob(os.path.join(UPLOAD_DIR, "*.pdf"))
    existing = [d["filename"] for d in db_manager.get_all_documents()]

    result = []
    for fpath in pdf_files:
        fname = os.path.basename(fpath)
        result.append({
            "name": fname,
            "size": os.path.getsize(fpath),
            "path": fpath,
            "in_db": fname in existing,
        })

    return {
        "files": result,
        "total": len(result),
        "unprocessed": len([r for r in result if not r["in_db"]]),
    }


@router.get("/existing-files/{filename:path}")
async def serve_existing_file(filename: str):
    """
    提供 data_reports 中的 PDF 文件内容
    """
    safe_name = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件 '{safe_name}' 不存在")
    return FileResponse(file_path, media_type="application/pdf", filename=safe_name)


# ──────────────────────────────────────────────
# 内置知识库文件上传 (.md / .txt / .jsonl)
# ──────────────────────────────────────────────

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "data", "finance_kb")


@router.post("/upload-kb")
async def upload_kb_file(file: UploadFile = File(...)):
    """上传内置知识库文件 (.md/.txt/.jsonl)，保存到 data/finance_kb/ 后由文件监听自动同步

    - .md / .txt: 直接保存，文件监听5秒后触发 sync_kb 全量同步
    - .jsonl: 识别类型后合并到对应文件（catalog.jsonl 或 kg_domain_triples.jsonl）
    """
    import shutil

    os.makedirs(KB_DIR, exist_ok=True)

    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".md", ".txt", ".jsonl"):
        raise HTTPException(status_code=400, detail="仅支持 .md / .txt / .jsonl 文件")

    safe_name = filename.replace(" ", "_").replace("/", "_")

    if ext == ".jsonl":
        content = await file.read()
        lines = [l for l in content.decode("utf-8").splitlines() if l.strip()]
        if not lines:
            raise HTTPException(status_code=400, detail="JSONL 文件为空")

        first = json.loads(lines[0])
        if "head" in first and "relation" in first:
            target = os.path.join(KB_DIR, "kg_domain_triples.jsonl")
        elif "id" in first and "name" in first:
            target = os.path.join(KB_DIR, "catalog.jsonl")
        else:
            target = os.path.join(KB_DIR, safe_name)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content.decode("utf-8"))
            return {"filename": safe_name, "status": "saved",
                    "note": "文件监听将在5秒后自动触发知识库同步"}

        # 合并去重
        existing_keys = set()
        if os.path.exists(target):
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        existing_keys.add(row.get("id") or
                                          (row.get("head"), row.get("relation"), row.get("tail")))

        added = 0
        with open(target, "a", encoding="utf-8") as f:
            for line in lines:
                row = json.loads(line)
                key = row.get("id") or (row.get("head"), row.get("relation"), row.get("tail"))
                if key not in existing_keys:
                    f.write(line + "\n")
                    existing_keys.add(key)
                    added += 1

        logger.info("知识库JSONL合并: file=%s target=%s added=%d",
                    safe_name, os.path.basename(target), added)
        return {"filename": safe_name, "status": "merged",
                "target": os.path.basename(target), "added": added,
                "note": "文件监听将在5秒后自动触发知识库同步"}
    else:
        dest = os.path.join(KB_DIR, safe_name)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("知识库文件已保存: %s", dest)
        return {"filename": safe_name, "status": "saved",
                "path": f"data/finance_kb/{safe_name}",
                "note": "文件监听将在5秒后自动触发知识库同步"}
