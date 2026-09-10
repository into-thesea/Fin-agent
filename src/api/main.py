"""
Fin-Agent FastAPI 应用 — 入口协调器

职责:
  - 应用生命周期管理 (lifespan)
  - CORS / 中间件
  - 路由注册
  - 前端静态文件托管 (单端口部署)
"""

import os
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.infra.logging_config import get_logger, set_trace_id
from src.infra.paths import PROJECT_ROOT
from src.exceptions import FinAgentError, error_response

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# 生命周期管理
# ──────────────────────────────────────────────

async def _warmup_model():
    """后台预热嵌入模型（bge-base-zh-v1.5 ~400MB, 约 3-5s 加载）

    注意: HybridRetriever() 构建是同步的 (BGE 模型加载),
    必须用 run_in_executor 避免阻塞事件循环.
    """
    try:
        logger.info("正在预热嵌入模型 (bge-base-zh-v1.5)...")

        def _do_warmup():
            from src.retrieval.retriever import HybridRetriever
            r = HybridRetriever()
            emb = r.model.encode(["预热"], normalize_embeddings=True)
            return emb.shape

        loop = asyncio.get_event_loop()
        shape = await loop.run_in_executor(None, _do_warmup)
        logger.info("嵌入模型预热完成 (维度: %s)", shape)
    except Exception as e:
        logger.warning("模型预热失败 (可忽略): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭事件"""
    logger.info("Fin-Agent API 服务启动中...")

    # 尝试启动 Redis Pub/Sub 监听器 (非阻塞，失败则跳过)
    from src.core.ws_manager import listener
    ws_task = None
    try:
        await asyncio.wait_for(listener.start(), timeout=3)
        if listener._running:
            ws_task = asyncio.create_task(listener.listen())
            logger.info("WebSocket 进度推送后台任务已启动")
        else:
            logger.info("Redis 不可用，WebSocket 推送降级 (仅轮询)")
    except asyncio.TimeoutError:
        logger.warning("Redis 连接超时，WebSocket 推送降级")
    except Exception as e:
        logger.warning("Redis 监听器启动失败: %s (WebSocket 降级)", e)

    # 预热 bge-base-zh-v1.5 (约 3-5s, 轻量不 OOM)
    asyncio.create_task(_warmup_model())

    yield  # 应用运行中

    # 关闭
    logger.info("Fin-Agent API 服务关闭中...")
    if ws_task:
        ws_task.cancel()
    try:
        await listener.stop()
    except Exception as e:
        logger.warning("监听器关闭异常 (可忽略): %s", e)
    # 清理嵌入模型引用, 释放内存
    try:
        from src.cache.faiss_manager import faiss_manager
        faiss_manager.reset()
    except Exception as e:
        logger.debug("faiss_manager 清理失败 (可忽略): %s", e)
    logger.info("Fin-Agent API 服务已完全关闭")


# ──────────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────────

app = FastAPI(
    title="Fin-Agent API",
    description="工业级图增强金融智能体 — 企业级 API",
    version="4.0.0",
    lifespan=lifespan,
)

# CORS — 显式列出允许源 (不能 allow_origins=* + credentials=True, 浏览器会拒绝)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",       # Vite dev server
        "http://localhost:8001",       # 生产模式 (单端口)
        "http://localhost:8000",       # 旧端口
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# 中间件
# ──────────────────────────────────────────────

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """为每个请求注入 trace_id + 记录监控指标 (P2)"""
    trace_id = request.headers.get("X-Trace-ID", f"req_{uuid.uuid4().hex[:12]}")
    set_trace_id(trace_id)
    start = time.time()

    response = await call_next(request)

    duration = round((time.time() - start) * 1000)
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Response-Time"] = f"{duration}ms"

    logger.info(
        "%s %s -> %d (%dms)",
        request.method, request.url.path, response.status_code, duration,
        extra={"method": request.method, "path": request.url.path,
               "status": response.status_code, "latency_ms": duration,
               "trace_id": trace_id},
    )

    # P2: 指标聚合 (Redis 原子计数 + 内存降级, 失败不影响响应)
    try:
        from src.monitor.metrics_store import metrics_store
        metrics_store.record(response.status_code, duration)
    except Exception as e:
        logger.debug("指标记录失败 (可忽略): %s", e)

    return response


# ── 全局异常处理器 ──────────────────────────────

@app.exception_handler(FinAgentError)
async def fin_agent_error_handler(request: Request, exc: FinAgentError):
    """FinAgentError → 统一 JSON 错误响应"""
    from src.infra.logging_config import get_trace_id
    trace_id = get_trace_id()
    logger.warning(
        "%s %s -> FinAgentError %s: %s",
        request.method, request.url.path, exc.code, exc.message,
        extra={"code": exc.code, "error_message": exc.message, "trace_id": trace_id},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc, trace_id),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """未预期的异常 → 500 统一 JSON 错误响应"""
    from src.infra.logging_config import get_trace_id
    trace_id = get_trace_id()
    logger.exception(
        "未捕获异常: %s %s -> %s",
        request.method, request.url.path, exc,
        extra={"trace_id": trace_id},
    )
    return JSONResponse(
        status_code=500,
        content=error_response(
            FinAgentError(
                message="服务器内部错误，请稍后重试",
                code="INTERNAL_ERROR",
                details={"exception_type": type(exc).__name__},
            ),
            trace_id,
        ),
    )


# ──────────────────────────────────────────────
# 路由注册
# ──────────────────────────────────────────────

from src.api.routes.chat import router as chat_router
from src.api.routes.auth import router as auth_router
from src.api.routes.knowledge import router as knowledge_router
from src.api.routes.tasks import router as tasks_router
from src.api.routes.monitor import router as monitor_router
from src.api.routes.monitor import health_router
from src.api.routes.handoff import router as handoff_router

app.include_router(chat_router)
app.include_router(auth_router)
app.include_router(knowledge_router)
app.include_router(tasks_router)
app.include_router(monitor_router)
app.include_router(health_router)  # /health, /health/ready
app.include_router(handoff_router)  # 转人工工单闭环

# WebSocket 路由
from src.core.ws_manager import ws_router
if ws_router:
    app.include_router(ws_router, prefix="")


# ──────────────────────────────────────────────
# 前端静态文件托管 (单端口部署)
# ──────────────────────────────────────────────

FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/favicon.svg")
    async def favicon():
        return FileResponse(os.path.join(FRONTEND_DIST, "favicon.svg"))

    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    @app.api_route("/{path_name:path}", methods=["GET"])
    async def serve_frontend(path_name: str):
        if path_name.startswith("api/") or path_name.startswith("ws/") or path_name.startswith("health"):
            raise HTTPException(status_code=404)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    logger.info("前端静态文件已加载: %s", FRONTEND_DIST)
else:
    logger.warning("前端构建目录不存在 (%s)，仅 API 模式运行", FRONTEND_DIST)


# ──────────────────────────────────────────────
# 直接运行
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
