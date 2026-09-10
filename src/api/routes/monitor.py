"""
Fin-Agent 监控与健康检查路由

提供:
  - GET /health               服务健康检查
  - GET /health/ready         就绪检查 (K8s Probe)
  - GET /api/v1/monitor/metrics  监控指标
"""

import logging
import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["监控"])

# 单独的健康检查前缀 — 挂载时无前缀
health_router = APIRouter(tags=["健康检查"])


@health_router.get("/health")
async def health_check():
    """服务健康检查 (动态检测各组件状态)"""
    redis_status = "degraded"
    try:
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        redis_status = "ok" if cache.ping() else "degraded"
    except Exception:
        redis_status = "down"

    return JSONResponse(content={
        "status": "ok",
        "version": "4.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "redis": redis_status,
        },
    })


@health_router.get("/health/ready")
async def readiness_check():
    """就绪检查 (K8s Probe)"""
    return {"status": "ready"}


@router.get("/api/v1/system/status")
async def system_status():
    """返回所有后端服务的健康状态 (异步并行检查)"""
    loop = asyncio.get_event_loop()

    async def _check_redis():
        from src.cache.redis_client import RedisCache
        try:
            cache = RedisCache()
            return "ok" if cache.ping() else "degraded"
        except Exception:
            return "down"

    async def _check_celery():
        from src.celery_app import is_celery_worker_running
        try:
            ok = await asyncio.wait_for(
                loop.run_in_executor(None, is_celery_worker_running, 1.0),
                timeout=2.0,
            )
            return "ok" if ok else "down"
        except (asyncio.TimeoutError, Exception):
            return "down"

    async def _check_faiss():
        try:
            from src.cache.faiss_manager import faiss_manager
            # 触发延迟加载后再读总数
            _ = faiss_manager.total_count
            if faiss_manager.is_loaded:
                return "ok" if faiss_manager.total_count > 0 else "empty"
            # 索引文件存在但未加载 → 尝试加载
            faiss_manager.reload()
            _ = faiss_manager.total_count
            return "ok" if faiss_manager.is_loaded and faiss_manager.total_count > 0 else "degraded"
        except Exception:
            return "degraded"

    async def _check_model():
        """检查嵌入模型 (bge-base-zh-v1.5) 是否已加载"""
        try:
            from src.retrieval.retriever import EMBEDDING_MODEL_LOADED
            return "ok" if EMBEDDING_MODEL_LOADED else "loading"
        except Exception:
            return "unknown"

    # 并行执行所有检查
    results = await asyncio.gather(
        _check_redis(),
        _check_celery(),
        _check_faiss(),
        _check_model(),
    )

    services = {
        "redis": results[0],
        "celery_worker": results[1],
        "faiss": results[2],
        "model": results[3],
    }

    overall = "ok" if all(v == "ok" for v in services.values() if v != "unknown") else "degraded"

    return {
        "status": overall,
        "services": services,
    }


@router.get("/api/v1/monitor/metrics")
async def get_metrics():
    """监控指标 (含客服: 意图分布 / 转人工率)"""
    from src.cache.redis_client import RedisCache

    cache = RedisCache()
    cache_stats = cache.get_stats()

    metrics = {
        "cache_hit_ratio": cache_stats.get("hit_ratio", 0),
        "cache_keys": cache_stats.get("total_keys", 0),
        "uptime_days": cache_stats.get("uptime_in_days", 0),
    }

    from src.database import db_manager
    db_stats = db_manager.get_stats()
    metrics.update({
        "documents_count": db_stats["documents"],
        "queries_count": db_stats["queries"],
        "chunks_count": db_stats["chunks"],
        # 客服指标
        "intent_stats": db_manager.get_intent_stats(),
        "handoff_count": db_manager.count_handoff_tickets(),
        "handoff_rate": round(
            db_manager.count_handoff_tickets() / max(db_stats["queries"], 1) * 100, 1
        ),
    })

    return metrics


@router.get("/api/v1/monitor/audit-logs")
async def get_audit_logs(limit: int = 20):
    """最近审计日志 (监控页)"""
    from src.database import db_manager
    return {"logs": db_manager.get_recent_audit_logs(limit)}


@router.get("/api/v1/monitor/sli")
async def get_sli(window: str = "1h"):
    """P2: 服务指标聚合 — QPS/成功率/错误率/超时率/缓存命中率/P50-P90-P99 延迟

    数据源: Redis 原子计数 (metrics_store), Redis 不可用时进程内存降级。
    趋势: window=1h 当前小时桶 / window=24h 最近 24 个桶聚合。
    """
    if window not in ("1h", "24h"):
        raise HTTPException(status_code=400, detail="window 仅支持 1h / 24h")
    from src.monitor.metrics_store import metrics_store
    return metrics_store.snapshot(window=window)


# ──────────────────────────────────────────────
# Badcase 闭环 (候选池 / triage / 提升为 golden)
# ──────────────────────────────────────────────

class TriagePatch(BaseModel):
    status: str = ""     # new | confirmed | fixed | ignored
    note: str = ""


class PromoteBody(BaseModel):
    note: str = ""


@router.get("/api/v1/badcases")
async def list_badcases(status: str = "", limit: int = 100, offset: int = 0):
    """badcase 候选列表 (条件在查询侧派生, 不落盘)"""
    from src.database import db_manager
    items = db_manager.query_badcases(status=status, limit=limit, offset=offset)
    return {"items": items, "limit": limit, "offset": offset}


@router.get("/api/v1/badcases/{trace_id}")
async def get_badcase(trace_id: str):
    """单条 trace 全文 (含答案与反馈)"""
    from src.database import db_manager
    trace = db_manager.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="trace 不存在")
    trace["reasons"] = db_manager._badcase_reasons(trace)
    return trace


@router.patch("/api/v1/badcases/{trace_id}")
async def patch_badcase(trace_id: str, body: TriagePatch):
    """更新 triage 状态 / 备注"""
    from src.database import db_manager
    ok = db_manager.update_triage(
        trace_id,
        status=body.status or None,
        note=body.note if body.note else None,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="trace 不存在或无可更新字段")
    return {"ok": True}


@router.post("/api/v1/badcases/{trace_id}/promote")
async def promote_badcase(trace_id: str, body: PromoteBody = None):
    """提升为 golden 骨架 (evidence 留空, 必须人工补齐)"""
    from src.database import db_manager
    from src.core.golden import promote_to_golden
    trace = db_manager.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="trace 不存在")
    note = body.note if body else ""
    out = promote_to_golden(trace, note=note)
    db_manager.update_triage(trace_id, status="confirmed", note=note or None)
    return out
