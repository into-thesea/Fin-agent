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

from celery import Task

from src.celery_app import celery_app
from src.cache.redis_client import RedisCache

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

    def _sync_db(self, status: str, progress: int, stage: str = "", error: str = ""):
        """把状态同步到 task_logs 表。

        这张表原先「有表有方法但无人写入」, 于是删除文档时的 Celery revoke
        永远查不到 task_id —— 撤销逻辑是死代码。进度是诊断信息, 写失败不应
        让 ETL 崩, 但必须留下 ERROR 日志而不是静默跳过。
        """
        try:
            from src.database import db_manager
            db_manager.update_task(self.task_id, status, progress=progress,
                                   stage=stage, error=error)
        except Exception as e:
            logger.error("[ETL] task_logs 状态回写失败 (%s): %s", self.task_id[:12], e)

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
        self._sync_db(status, progress, stage, error)

        # Celery 内置状态 (给 flower 等监控工具)
        self.task.update_state(
            state=status.upper(),
            meta={
                "progress": progress,
                "stage": stage,
                "error": error,
            },
        )

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
        self._sync_db("completed", 100, "completed")
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
        self._sync_db("failed", self._step, "failed", error)
        self.task.update_state(
            state="FAILURE",
            meta={"progress": self._step, "stage": "failed", "error": error},
        )


# ──────────────────────────────────────────────
# ETL 任务定义
# ──────────────────────────────────────────────

@celery_app.task(bind=True, name="etl_pipeline")
def etl_pipeline_task(self: Task, pdf_path: str, filename: str = None, version_tag: str = ""):
    """
    异步 ETL 流水线 —— 只是 `pipeline_manager.ingest_document` 的 Celery 外壳。

    实现只有一处 (src/core/pipeline_manager.py)。原先这里另有一整套
    dedup/parse/vector/db 流程, 用的是**按页切分**且**不重建 BM25**;
    同步降级路径用的是章节感知分块且重建 BM25 —— 同一份 PDF 走两条路进去,
    产出完全不同的分块和索引覆盖。

    刻意不自动重试: 流水线是「先写索引(Milvus+BM25)再写库」, 不是幂等的,
    重试会重复追加向量。失败要看得见, 而不是靠重试掩盖。
    (原先还有 `reporter.fail()` 在 `self.retry()` 之前 —— 重试期间前端就判死了,
     最终任务成功但界面永久显示失败。)

    Args:
        pdf_path: PDF 文件的绝对路径
        filename: 文件名 (显示用)
        version_tag: 版本标签

    Returns:
        {"status": "success"|"skipped"|"failed", "doc_id": int, ...}
    """
    from src.core.pipeline_manager import ingest_document

    # 规范化路径 (Windows 反斜杠在 JSON 序列化中会被错误转义)
    pdf_path = os.path.normpath(pdf_path)
    file_name = filename or os.path.basename(pdf_path)
    reporter = ProgressReporter(self, total_steps=4)
    start_time = time.time()

    try:
        result = ingest_document(
            pdf_path,
            source_label="etl_async",
            version_tag=version_tag,
            task_id_for_progress=self.request.id,  # 进度写 Redis, 供前端轮询/WS
        )
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("[ETL] 流水线崩溃 (%s): %s", file_name, error_msg, exc_info=True)
        reporter.fail(error_msg)
        raise  # 让 Celery 如实记录 FAILURE, 不伪装成正常返回

    status = result.get("status")
    if status in ("success", "skipped"):
        reporter.complete(result.get("doc_id", 0), result.get("duration", 0))
    else:
        error_msg = result.get("error", "unknown")
        logger.error("[ETL] 流水线失败 (%s): %s", file_name, error_msg)
        reporter.fail(error_msg)

    return result
