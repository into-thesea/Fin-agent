"""
Fin-Agent 任务状态路由

提供:
  - GET /api/v1/tasks/{task_id}  查询异步 ETL 任务状态
"""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["任务"])

# 全系统统一的任务状态取值 —— 写入侧(ProgressReporter / pipeline_manager)与
# 前端 FileItemStatus 都以此为准。
# Celery 自带的大写状态要映射过来, 否则它直接 lower() 出来是 "success",
# 不在前端认的四个值里(前端只认 completed), 页面会卡在「进行中」。
TASK_STATUS = ("pending", "processing", "completed", "failed")
_CELERY_STATE_MAP = {
    "PENDING": "pending",
    "RECEIVED": "pending",
    "STARTED": "processing",
    "PROGRESS": "processing",
    "RETRY": "processing",
    "SUCCESS": "completed",
    "FAILURE": "failed",
    "REVOKED": "failed",
}


def _normalize_state(state) -> str:
    """把任意来源的状态规范到 TASK_STATUS；未知值按 failed 处理。

    未知值不能原样透传 —— 前端拿到不认识的状态会一直显示「进行中」,
    比直接显示失败更难排查。
    """
    raw = str(state or "").strip()
    if raw.lower() in TASK_STATUS:
        return raw.lower()
    mapped = _CELERY_STATE_MAP.get(raw.upper())
    if mapped:
        return mapped
    logger.error("未知任务状态 %r, 按 failed 处理", raw)
    return "failed"


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """查询异步 ETL 任务状态

    状态源: Redis (实时进度) → Celery backend (权威终态) → task_logs 表 (历史)。

    全局唯一, 不再有进程内存兜底 —— 内存字典在多 uvicorn worker 下轮询打到
    别的 worker 会直接 404, 给出一个「可能对可能错」的状态比明确报 503 更糟。
    """
    # 1. Redis 实时进度
    from src.cache.redis_client import RedisCache

    cache = RedisCache()
    status = cache.get_task_status(task_id)

    # 2. task_logs 表 (Redis TTL 过期后仍可查; ETL worker 全程回写)
    if not status:
        try:
            from src.database import db_manager
            row = db_manager.get_task(task_id)
            if row:
                return {
                    "task_id": row["task_id"],
                    "status": _normalize_state(row["status"]),
                    "progress": row["progress"] or 0,
                    "stage": row["stage"] or "",
                    "error": row["error"] or "",
                }
        except Exception as e:
            logger.error("task_logs 查询失败 (%s): %s", task_id[:12], e)

    # 3. 尝试从 Celery 后端查询 (异步模式)
    if not status:
        try:
            from src.celery_app import celery_app
            result = celery_app.AsyncResult(task_id)
            # 如果任务 ID 格式不对或 Redis 不可达，
            # AsyncResult 可能立即失败或 state 访问失败
            try:
                state = result.state
            except Exception as e:
                logger.warning("任务 %s: AsyncResult.state 失败 (%s: %s)", task_id, type(e).__name__, str(e)[:100])
                raise HTTPException(
                    status_code=503,
                    detail="任务状态查询暂时不可用 (Redis 未连接)",
                )

            if state == "PENDING":
                raise HTTPException(status_code=404, detail="未知任务 ID")

            status = {
                "task_id": task_id,
                "status": _normalize_state(state),
                "progress": result.meta.get("progress", 0) if result.meta else 0,
                "stage": result.meta.get("stage", "") if result.meta else "",
                "error": str(result.info) if result.failed() else "",
            }
        except HTTPException:
            raise  # 直接传递 404/503
        except Exception as e:
            logger.warning("任务 %s 状态查询异常 (%s: %s)", task_id, type(e).__name__, str(e)[:100])
            raise HTTPException(
                status_code=503,
                detail="任务状态查询暂时不可用 (Redis 未连接)",
            )

    return status
