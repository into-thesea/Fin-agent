"""
Fin-Agent 异步 ETL 流水线任务

每个阶段自动上报进度至 Redis:
  - task:{task_id}:status -> "processing" | "completed" | "failed"
  - task:{task_id}:progress -> 0-100
  - task:{task_id}:stage -> "parsing" | "vectorizing" | "extracting" | "graph_merge"
  - task:{task_id}:error -> error message (if failed)

前端可通过 WebSocket 或轮询 Redis 获取实时进度。
"""

import os
import sys
import time
import logging
import traceback
from typing import Optional

from celery import Task
from celery.exceptions import MaxRetriesExceededError

from src.celery_app import celery_app
from src.cache.redis_client import RedisCache
from src.core.ws_manager import publish_progress

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)


# ──────────────────────────────────────────────
# 进度上报工具
# ──────────────────────────────────────────────

class ProgressReporter:
    """
    ETL 进度上报器

    通过 Redis 发布任务进度，供 WebSocket 网关或轮询接口消费。
    同时支持 Celery 内置的 task.update_state() 作为 fallback。
    """

    def __init__(self, task: Task, total_steps: int = 7):
        self.task = task
        self.task_id = task.request.id
        self.total_steps = total_steps
        self.cache = RedisCache()
        self._step = 0

    def report(self, stage: str, step: int = None, status: str = "processing", error: str = ""):
        """上报当前进度"""
        if step is not None:
            self._step = step
        else:
            self._step += 1

        progress = min(int(self._step / self.total_steps * 100), 99)

        # 写入 Redis (给 WebSocket 消费)
        self.cache.set_task_status(
            task_id=self.task_id,
            status=status,
            progress=progress,
            stage=stage,
            error=error,
        )

        # Celery 内置状态 (给 flower 等监控工具)
        self.task.update_state(
            state=status.upper(),
            meta={
                "progress": progress,
                "stage": stage,
                "error": error,
            },
        )

        # Redis Pub/Sub (给 WebSocket 客户端)
        publish_progress(self.task_id, status, progress, stage, error)

        logger.info(
            "[ETL] %s: %s (%d%%)",
            self.task_id[:12], stage, progress,
            extra={"task_id": self.task_id, "stage": stage, "progress": progress},
        )

    def complete(self, doc_id: int, duration: float):
        """标记任务完成"""
        self.cache.set_task_status(
            task_id=self.task_id,
            status="completed",
            progress=100,
            stage="completed",
        )
        self.task.update_state(
            state="SUCCESS",
            meta={"progress": 100, "stage": "completed", "doc_id": doc_id, "duration": duration},
        )

    def fail(self, error: str):
        """标记任务失败"""
        self.cache.set_task_status(
            task_id=self.task_id,
            status="failed",
            progress=self._step,
            stage="failed",
            error=error,
        )
        self.task.update_state(
            state="FAILURE",
            meta={"progress": self._step, "stage": "failed", "error": error},
        )


# ──────────────────────────────────────────────
# ETL 任务定义
# ──────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="etl_pipeline",
    max_retries=3,
    default_retry_delay=60,  # 1 分钟后重试
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # 最大重试间隔 10 分钟
    retry_jitter=True,
)
def etl_pipeline_task(self: Task, pdf_path: str, filename: str = None, version_tag: str = ""):
    """
    异步 ETL 流水线: PDF → 分块 → 向量化 → 实体提取 → 图谱注入

    这是 Celery 任务，执行 `celery -A src.celery_app worker` 后自动可用。

    Args:
        pdf_path: PDF 文件的绝对路径
        filename: 文件名 (显示用)
        version_tag: 版本标签

    Returns:
        {"status": "success", "doc_id": int, "duration": float, ...}
    """
    # 延迟导入 (避免 Worker 启动时加载所有 ML 模型)
    from src.main import extract_chunks_from_pdf
    from src.retrieval.vector_indexer import add_chunks_to_index
    from src.database import db_manager

    # 规范化路径 (Windows 反斜杠在 JSON 序列化中会被错误转义)
    pdf_path = os.path.normpath(pdf_path)
    reporter = ProgressReporter(self, total_steps=4)
    file_name = filename or os.path.basename(pdf_path)
    start_time = time.time()

    if not version_tag:
        version_tag = f"etl_{time.strftime('%Y%m%d_%H%M%S')}"

    try:
        # ── Step 1: 内容寻址去重 ──
        reporter.report("dedup", step=0)
        content_hash = db_manager.compute_file_hash(pdf_path)
        existing = db_manager.find_document_by_hash(content_hash)
        if existing:
            reporter.complete(existing["id"], 0)
            return {
                "status": "skipped",
                "message": f"Content already exists (ID: {existing['id']})",
                "doc_id": existing["id"],
                "content_hash": content_hash,
            }

        # ── Step 2: PDF 解析 ──
        reporter.report("parsing", step=1)
        chunks = extract_chunks_from_pdf(pdf_path)
        if not chunks:
            raise ValueError("No content extracted from PDF")
        logger.info("[ETL] 解析完成: %d chunks", len(chunks))

        # ── Step 3: 向量索引 ──
        reporter.report("vectorizing", step=2)
        added = add_chunks_to_index(chunks, document_id=file_name)
        logger.info("[ETL] 向量化完成: %d 条", added)

        # ── Step 4: 数据库记录 ──
        reporter.report("db_record", step=3)
        file_size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        doc_id = db_manager.add_document(
            filename=file_name,
            status="done",
            count=len(chunks),
            content_hash=content_hash,
            file_size=file_size,
            source_label="etl_async",
        )

        # ── Step 5: 缓存失效 ──
        reporter.report("cache_invalidation", step=4)
        cache = RedisCache()
        cache.invalidate_document_cache(file_name)

        duration = round(time.time() - start_time, 2)
        reporter.complete(doc_id, duration)

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks_count": len(chunks),
            "duration": duration,
            "content_hash": content_hash,
            "version_tag": version_tag,
        }

    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ETL] 流水线失败 (%s): %s", file_name, error_msg, exc_info=True)

        reporter.fail(error_msg)

        # 自动重试
        try:
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            logger.error("[ETL] 重试耗尽: %s", file_name)
            return {
                "status": "failed",
                "error": error_msg,
                "duration": duration,
            }
