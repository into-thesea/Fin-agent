"""
Fin-Agent 监控与健康检查路由

提供:
  - GET /health               服务健康检查
  - GET /health/ready         就绪检查 (K8s Probe)
  - GET /api/v1/monitor/metrics  监控指标
"""

import asyncio
import json
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.routes.auth import require_roles
from src.infra.paths import DATA_DIR

logger = logging.getLogger(__name__)

# 整组鉴权: 本组包含审计日志(用户原始提问)、问题复盘明细、评测数据,
# 还有 badcases 的写接口 —— 这些都不该匿名可读可写。
# knowledge.py / handoff.py 一直挂着守卫, 唯独这里漏了(实测无 token 也返回 200)。
router = APIRouter(tags=["监控"],
                   dependencies=[Depends(require_roles("admin", "analyst"))])

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
    """就绪检查 (K8s Readiness Probe)

    嵌入模型完成后台预热(import torch/transformers ~11s + 权重加载)后才算就绪。
    未就绪返回 503, 让负载均衡/部署暂不把流量打进来 —— 冷启动成本因此不落在
    任何真实用户身上; 即使预热窗口内有请求, Python import 锁与模型双检锁
    也保证重型依赖只加载一次, 请求是等待而非重复触发。
    """
    try:
        from src.retrieval.retriever import EMBEDDING_MODEL_LOADED
        model_ready = bool(EMBEDDING_MODEL_LOADED)
    except Exception:
        model_ready = False
    if model_ready:
        return JSONResponse({"status": "ready", "embedding_model": "loaded"},
                            status_code=200)
    return JSONResponse({"status": "not_ready", "embedding_model": "warming"},
                        status_code=503)


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
                loop.run_in_executor(None, is_celery_worker_running),
                timeout=2.0,
            )
            return "ok" if ok else "down"
        except (asyncio.TimeoutError, Exception):
            return "down"

    async def _check_milvus():
        """Milvus 是唯一向量检索后端 —— 它挂了检索就是不可用, 不再降级 FAISS"""
        def _probe():
            from src.vectorstore.milvus_manager import MilvusManager
            mgr = MilvusManager()
            if not mgr.available:
                return "down"
            return "ok" if mgr.total_count > 0 else "empty"
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, _probe), timeout=5.0)
        except Exception:
            return "down"

    async def _check_neo4j():
        """Neo4j 是图谱检索的唯一后端 —— 它挂了关系类查询会直接报错"""
        def _probe():
            from src.knowledge_graph.core import Neo4jGraph
            return "ok" if Neo4jGraph().available else "down"
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, _probe), timeout=5.0)
        except Exception:
            return "down"

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
        _check_milvus(),
        _check_neo4j(),
        _check_model(),
    )

    services = {
        "redis": results[0],
        "celery_worker": results[1],
        "milvus": results[2],      # 向量检索后端 (唯一)
        "neo4j": results[3],       # 图谱检索后端 (唯一)
        "model": results[4],
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
    from src.core.golden import promote_to_golden
    from src.database import db_manager
    trace = db_manager.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="trace 不存在")
    note = body.note if body else ""
    out = promote_to_golden(trace, note=note)
    db_manager.update_triage(trace_id, status="confirmed", note=note or None)
    return out


# ──────────────────────────────────────────────
# 评测看板 (只读)
# ──────────────────────────────────────────────

EVAL_DIR = os.path.join(DATA_DIR, "eval")
HISTORY_PATH = os.path.join(EVAL_DIR, "history.jsonl")
REPORT_PATH = os.path.join(EVAL_DIR, "report_finance.json")

# 评测脚本产出的指标名 → 中文标签 (前端展示用, 后端统一给出避免两边各写一份)
METRIC_LABELS = {
    "evidence_coverage": "证据覆盖",
    "path_structure_consistency": "路径一致",
    "graph_trigger_ratio": "图谱触发",
}


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 截断行跳过, 与其它 JSONL 读取保持一致
    return out


def _dedupe_by_version(rows: list) -> list:
    """同一知识库版本只保留最后一次快照。

    history.jsonl 是「每次同步追加一条」, 同一版本常被重复评测多次
    (实测 14 条里有 9 条是同一个 version 的相同数值)。不去重的话版本趋势图
    会变成一堆重复点的平线, 看不出任何变化。
    """
    by_version: dict = {}
    for r in rows:
        v = r.get("version") or "?"
        by_version[v] = r          # 后者覆盖前者 = 保留该版本最后一次
    return sorted(by_version.values(), key=lambda r: r.get("at", ""))


@router.get("/api/v1/eval/summary")
async def eval_summary():
    """评测看板数据源: 当前指标 + 逐条明细 + 版本趋势

    直接读评测脚本的产物文件, 不额外建表 —— history.jsonl 由
    scripts/sync_kb.py 每次同步后追加, report_finance.json 由
    scripts/eval_finance.py 产出。

    缺失的段如实标 null / available=false, 让前端显示「评测未运行」,
    而不是渲染一份看起来正常、实际缺一半的报告。
    """
    def _load():
        rows = _read_jsonl(HISTORY_PATH)
        history = _dedupe_by_version(rows)

        report = None
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
        return rows, history, report

    rows, history, report = await asyncio.get_event_loop().run_in_executor(None, _load)

    current = None
    if report:
        offline = report.get("offline") or {}
        current = {
            "version": report.get("version", ""),
            "golden_n": report.get("golden_n", 0),
            "metrics": {k: offline.get(k) for k in METRIC_LABELS},
            "n": offline.get("n", 0),
            "rows": offline.get("rows", []),
            # 在线评测(LLM-as-judge)当前未纳入, 如实标出而不是省略
            "online": report.get("online"),
        }

    return {
        "current": current,
        "history": history,
        "metric_labels": METRIC_LABELS,
        "available": {
            "history": bool(history),
            "report": report is not None,
            "online": bool(report and report.get("online")),
        },
        "raw_history_count": len(rows),
    }
