"""
Fin-Agent 任务状态路由

提供:
  - GET /api/v1/tasks/{task_id}  查询异步 ETL 任务状态
"""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["任务"])


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """查询异步 ETL 任务状态"""
    # 1. 尝试从 Redis 获取 (同步模式或主动写入)
    from src.cache.redis_client import RedisCache

    cache = RedisCache()
    status = cache.get_task_status(task_id)

    # 2. 在内存中查找同步任务兜底 (Redis no-op 时使用)
    if not status:
        try:
            from src.api.routes.knowledge import _SYNC_TASK_RESULTS
            in_mem = _SYNC_TASK_RESULTS.get(task_id)
            if in_mem:
                return in_mem
        except ImportError:
            pass  # knowledge 模块尚未加载，跳过

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
                "status": state.lower(),
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
