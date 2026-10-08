"""
Fin-Agent FastAPI 应用 — 入口协调器

职责:
  - 应用生命周期管理 (lifespan)
  - CORS / 中间件
  - 路由注册
  - 前端静态文件托管 (单端口部署)
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.exceptions import FinAgentError, error_response
from src.infra.logging_config import get_logger, set_trace_id
from src.infra.paths import PROJECT_ROOT

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


async def _start_kb_watcher():
    """启动知识库文件监听 — data/finance_kb/ 有变化自动同步（防抖5秒）"""
    import threading
    import time

    KB_DIR = os.path.join(PROJECT_ROOT, "data", "finance_kb")
    if not os.path.exists(KB_DIR):
        logger.warning("知识库目录不存在，跳过文件监听: %s", KB_DIR)
        return

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        logger.warning("watchdog 未安装，跳过知识库文件监听。安装: pip install watchdog")
        return

    _debounce = {"timer": None}

    def do_sync():
        logger.info("[KB监听] 检测到文件变化，开始同步...")
        try:
            import sys
            sync_path = os.path.join(PROJECT_ROOT, "scripts")
            if sync_path not in sys.path:
                sys.path.insert(0, sync_path)
            import sync_kb
            code = sync_kb.main()
            logger.info("[KB监听] 同步完成 (code=%s)", code)
        except Exception as e:
            logger.error("[KB监听] 同步失败: %s", e, exc_info=True)

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory:
                return
            if not event.src_path.endswith((".md", ".jsonl")):
                return
            if _debounce["timer"]:
                _debounce["timer"].cancel()
            _debounce["timer"] = threading.Timer(5.0, do_sync)
            _debounce["timer"].start()

    def run_watcher():
        observer = Observer()
        observer.schedule(Handler(), KB_DIR, recursive=False)
        observer.start()
        logger.info("[KB监听] 已启动，监听目录: %s", KB_DIR)
        try:
            while True:
                time.sleep(1)
        except Exception:
            observer.stop()
        observer.join()

    thread = threading.Thread(target=run_watcher, daemon=True, name="kb-watcher")
    thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭事件"""
    logger.info("Fin-Agent API 服务启动中...")

    # 预热 bge-base-zh-v1.5 (约 3-5s, 轻量不 OOM)
    asyncio.create_task(_warmup_model())
    # 启动知识库文件监听（data/finance_kb/ 变化自动同步）
    asyncio.create_task(_start_kb_watcher())

    yield  # 应用运行中

    # 关闭
    logger.info("Fin-Agent API 服务关闭中...")
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
    # SSE 不走这里: 此刻 duration 只是「响应对象返回」的耗时(实测 ~1ms), 不是用户感知的
    # 首 token 时间 —— 混进全局延迟样本会把 p99 拉成假象。流式自己在 chat.py 里按
    # 首个 token / 流结束打点 (metrics_store.record_stream)。
    # 判据必须用 content-type: BaseHTTPMiddleware 的 call_next 永远返回 _StreamingResponse
    # (Response 的子类), 用 isinstance(resp, StreamingResponse) 判断会静默失效。
    try:
        from src.monitor.metrics_store import metrics_store
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            logger.debug("SSE 请求不进全局延迟采样: %s", request.url.path)
        else:
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

from src.api.routes.auth import router as auth_router
from src.api.routes.chat import router as chat_router
from src.api.routes.handoff import router as handoff_router
from src.api.routes.kb_console import router as kb_console_router
from src.api.routes.knowledge import router as knowledge_router
from src.api.routes.monitor import health_router
from src.api.routes.monitor import router as monitor_router
from src.api.routes.tasks import router as tasks_router

app.include_router(chat_router)
app.include_router(auth_router)
app.include_router(knowledge_router)
app.include_router(tasks_router)
app.include_router(monitor_router)
app.include_router(health_router)  # /health, /health/ready
app.include_router(handoff_router)  # 转人工工单闭环
app.include_router(kb_console_router)  # 知识库管理台 (策略配置)

# 注: ETL 进度曾有一条 WebSocket 通道(/ws/task/{id})，2026-09-14 移除 ——
# ETL 只有 4~5 个阶段点, 3 秒轮询完全够用; 而 WS 每条上传任务要建一条连接
# (批量 20 个文件就是 20 条)、依赖 API 进程内的 Redis pub/sub 监听线程、
# 端点还缺鉴权。现在进度只走 GET /api/v1/tasks/{task_id}。


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
