"""
Fin-Agent 知识库路由 (文档管理 + ETL 任务追踪)

提供:
  - POST /api/v1/knowledge/upload             异步上传 PDF（单文件）
  - POST /api/v1/knowledge/batch-upload       异步批量上传 PDF（多文件）
  - DELETE /api/v1/knowledge/documents/{id}   删除文档（含 SQLite/FAISS/PDF/Celery）
  - GET  /api/v1/knowledge/documents          文档列表
  - GET  /api/v1/knowledge/stats              知识库统计
  - GET  /api/v1/tasks/{task_id}              查询异步任务状态
"""

import os
import logging
import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse

from src.api.routes.auth import require_roles

logger = logging.getLogger(__name__)

# 后端强制鉴权: 知识库管理仅管理员可操作
router = APIRouter(prefix="/api/v1/knowledge", tags=["知识库"],
                   dependencies=[Depends(require_roles("admin"))])

# 同步 I/O 辅助: 在默认线程池中执行, 不阻塞事件循环
async def _aio(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

# 同步任务结果内存兜底 (Redis no-op 时使用)
_SYNC_TASK_RESULTS: dict[str, dict] = {}

# 上传配置
from src.infra.paths import DATA_REPORTS_DIR
UPLOAD_DIR = DATA_REPORTS_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


# ──────────────────────────────────────────────
# 单文件上传
# ──────────────────────────────────────────────


async def _process_upload(file: UploadFile) -> dict:
    """处理单个文件上传：校验 → 存储 → 去重 → 提交 ETL"""
    if not file.filename or not file.filename.endswith(".pdf"):
        return {"filename": file.filename or "unknown", "status": "rejected", "message": "仅支持 PDF 文件"}

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return {"filename": file.filename, "status": "rejected", "message": "文件大小超过 50MB 限制"}

    safe_name = file.filename.replace(" ", "_").replace("/", "_")
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as f:
        f.write(content)

    from src.database import db_manager
    content_hash = db_manager.compute_file_hash(file_path)
    existing = db_manager.find_document_by_hash(content_hash)

    if existing:
        os.remove(file_path)
        logger.info("文件重复 (hash=%s): %s", content_hash[:12], safe_name)
        return {
            "filename": safe_name,
            "status": "skipped",
            "message": "内容已存在，跳过入库",
            "doc_id": existing["id"],
            "content_hash": content_hash,
        }

    # ── 检测 Celery Worker；不可用时降级到同步 ETL ──
    from src.celery_app import is_celery_worker_running
    version_tag = f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if is_celery_worker_running():
        # 异步路径: Celery Worker 在线
        from src.tasks.etl_tasks import etl_pipeline_task

        task = etl_pipeline_task.delay(
            pdf_path=file_path.replace("\\", "/"),
            filename=safe_name,
            version_tag=version_tag,
        )

        logger.info(
            "上传任务已提交 (异步): file=%s, task_id=%s",
            safe_name, task.id,
            extra={"task_id": task.id, "file": safe_name},
        )
        return {
            "filename": safe_name,
            "status": "accepted",
            "task_id": task.id,
            "content_hash": content_hash,
            "sync_mode": False,
        }
    else:
        # 同步降级: 后台线程跑 ETL，API 立即返回 (fire-and-forget)
        import uuid
        import threading
        sync_task_id = f"sync_{uuid.uuid4().hex[:12]}"
        logger.info("Celery Worker 不可用, 切换同步模式 (后台): file=%s, task_id=%s", safe_name, sync_task_id)

        # 写入初始状态到 Redis + 内存兜底 (供前端轮询)
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        cache.set_task_status(
            task_id=sync_task_id,
            status="processing",
            progress=0,
            stage="starting",
        )
        _SYNC_TASK_RESULTS[sync_task_id] = {
            "task_id": sync_task_id,
            "status": "processing",
            "progress": 0,
            "stage": "starting",
        }

        # 后台线程运行 ETL，不阻塞 API 响应
        from src.core.pipeline_manager import ingest_document

        def _run_sync_etl():
            """在后台线程中运行 ETL 并更新状态"""
            try:
                result = ingest_document(
                    pdf_path=file_path.replace("\\", "/"),
                    source_label="upload_sync",
                    version_tag=version_tag,
                    task_id_for_progress=sync_task_id,
                )
                status = result.get("status", "failed")
                if status in ("success", "skipped"):
                    cache.set_task_status(
                        task_id=sync_task_id,
                        status="completed",
                        progress=100,
                        stage="completed",
                    )
                else:
                    cache.set_task_status(
                        task_id=sync_task_id,
                        status="failed",
                        progress=0,
                        stage="failed",
                        error=result.get("error", "ETL 流水线执行失败"),
                    )
                _SYNC_TASK_RESULTS[sync_task_id] = {
                    "task_id": sync_task_id,
                    "status": "completed" if status in ("success", "skipped") else "failed",
                    "progress": 100 if status in ("success", "skipped") else 0,
                    "stage": "completed" if status in ("success", "skipped") else "failed",
                    "error": result.get("error", "") if status != "success" else "",
                }
            except Exception as e:
                logger.error("同步 ETL 失败 (%s): %s", safe_name, e, exc_info=True)
                cache.set_task_status(
                    task_id=sync_task_id,
                    status="failed",
                    progress=0,
                    stage="failed",
                    error=str(e)[:200],
                )
                _SYNC_TASK_RESULTS[sync_task_id] = {
                    "task_id": sync_task_id,
                    "status": "failed",
                    "progress": 0,
                    "stage": "failed",
                    "error": str(e)[:200],
                }

        thread = threading.Thread(target=_run_sync_etl, daemon=True)
        thread.start()

        return {
            "filename": safe_name,
            "status": "accepted",
            "task_id": sync_task_id,
            "content_hash": content_hash,
            "sync_mode": True,
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


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """
    删除文档及其所有关联数据

    清理项:
      1. SQLite 元数据库记录
      2. PDF 源文件
      3. FAISS 向量索引（逻辑删除）
      4. Celery 任务撤销（如果仍在运行）

    Returns: {"deleted": bool, "message": str, "details": dict}
    """
    from src.database import db_manager

    # 除 HTTPException 外的所有 I/O 操作在线程中执行
    def _delete_sync():
        doc = db_manager.get_document_by_id(doc_id)
        if not doc:
            return {"_404": True}

        details = {"sqlite": False, "pdf": False, "faiss": False, "celery": False}

        # 1. SQLite 删除
        details["sqlite"] = db_manager.delete_document(doc_id)

        # 2. PDF 源文件删除
        pdf_path = None
        filename = doc.get("filename", "")
        for candidate in [filename, filename.replace(" ", "_"), os.path.join(UPLOAD_DIR, filename)]:
            full_path = os.path.join(UPLOAD_DIR, os.path.basename(candidate))
            if os.path.exists(full_path):
                pdf_path = full_path
                break
        if pdf_path:
            try:
                os.remove(pdf_path)
                details["pdf"] = True
                logger.info("已删除 PDF: %s", pdf_path)
            except OSError as e:
                logger.warning("PDF 删除失败 %s: %s", pdf_path, e)

        # 3. FAISS 向量删除
        try:
            from src.cache.faiss_manager import faiss_manager
            count = faiss_manager.delete_by_document(str(doc_id))
            details["faiss"] = count > 0
        except Exception as e:
            logger.warning("FAISS 删除异常: %s", e)

        # 4. Celery 任务撤销
        try:
            from src.celery_app import celery_app
            cursor = db_manager.conn.cursor()
            cursor.execute(
                "SELECT task_id FROM task_logs WHERE filename = ? AND status NOT IN ('completed', 'failed')",
                (filename,),
            )
            task_row = cursor.fetchone()
            if task_row and task_row["task_id"]:
                celery_app.control.revoke(task_row["task_id"], terminate=True, signal='SIGTERM')
                details["celery"] = True
        except Exception as e:
            logger.warning("Celery 撤销异常: %s", e)

        return {"_404": False, "details": details, "filename": doc.get("filename", "")}

    result = await _aio(_delete_sync)
    if result.get("_404"):
        raise HTTPException(status_code=404, detail=f"文档 ID {doc_id} 不存在")

    details = result["details"]
    filename = result["filename"]
    success = details["sqlite"]
    return {
        "deleted": success,
        "message": f"文档 '{filename}' 删除成功" if success else "文档删除失败",
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
    from src.database import db_manager
    import glob

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
