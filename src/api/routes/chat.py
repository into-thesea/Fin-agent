"""
Fin-Agent 聊天路由 (v3.1 — Async LLM I/O + 安全后台任务)

核心改动:
  1. LLM HTTP 调用使用原生 async (httpx.AsyncClient), 不占线程
  2. 快速通道 (greeting/simple_fact): await chat_async(), 零线程消耗
  3. 复杂查询: WorkerPool (固定 2 线程), 不会耗尽系统默认线程池
  4. 后台任务使用 run_in_executor + ensure_future, 不阻塞事件循环
  5. dialog_state._lock 已改为 RLock 避免 update()→get_or_create() 死锁

API:
  POST /api/v1/chat/sync     同步问答 (带超时)
  POST /api/v1/chat/stream   SSE 流式问答
"""

import json
import asyncio
import logging
import time
import uuid
import threading
from functools import partial
from queue import Queue, Empty as QueueEmpty
from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.infra.logging_config import get_trace_id_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["聊天"])

# ⏱ 超时配置
SYNC_TIMEOUT = 120       # 同步接口超时 (秒)
STREAM_TIMEOUT = 150     # 流式接口超时 (秒)

class ChatRequest(BaseModel):
    query: str
    session_id: str = ""
    user_id: str = ""   # 当前用户 ID (订单归属校验用; 也可通过 X-User-Id header 传入)


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = 0          # 1 赞 / -1 踩
    comment: str = ""
    user_id: str = ""
    trace_id: str = ""       # 关联到具体轮次 (badcase 闭环)


def _apply_compliance(result: dict) -> dict:
    """出站合规: 掩码敏感词/不当承诺, 附 compliance 字段供前端提示"""
    ans = result.get("answer")
    if not ans:
        return result
    from src.core.compliance import check, finalize
    c = check(ans)
    final_text = finalize(ans)
    if final_text != ans:
        result["answer"] = final_text
    if not c["clean"] or c.get("pii"):
        result["compliance"] = {"clean": False, "hits": c["hits"], "pii": c.get("pii")}
    return result


def _create_handoff_ticket(session_id: str, user_id: str, reason: str = "") -> int:
    """创建人工转接工单 (供坐席队列), 返回 ticket_id"""
    try:
        from src.db.manager import db_manager
        tid = db_manager.add_handoff_ticket(session_id, user_id or "anonymous", reason[:200])
        logger.info("已创建转人工工单 #%s: session=%s", tid, session_id[:8])
        return tid
    except Exception as e:
        logger.debug("转人工工单创建失败 (可忽略): %s", e)
        return 0


# ── 同步接口 ──────────────────────────────────

@router.post("/sync")
async def chat_sync(req: ChatRequest, authorization: Optional[str] = Header(None)):
    """
    同步聊天接口 — async LLM I/O, 只在必要操作使用线程.

    流程:
      1. 状态/重写/缓存查找 (同步, 毫秒级)
      2. 缓存命中 → 直接返回
      3. 规则路由 → greeting/simple_fact → await chat_async() (零线程)
      4. 复杂查询 → WorkerPool.run_analysis_async() (限 2 线程)
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    original_query = req.query.strip()

    # 解析当前用户: 请求体 user_id → Bearer token (用于订单归属校验)
    user_id = req.user_id
    if not user_id:
        from src.api.routes.auth import resolve_user_id
        user_id = resolve_user_id(authorization) or ""

    # ── 1. 初始化 (同步, 毫秒级: 状态 + 重写 + 缓存) ──
    from src.core.dialog_state import get_state_manager
    from src.llm.query_rewriter import rewrite_query
    from src.core.answer_cache import get_cached as cache_get, set_cache as cache_set
    from src.core.fast_path import route_class
    from src.llm.llm_client import create_client
    from src.agents.prompts import BOUNDARY_BLOCK_LIGHT

    state_mgr = get_state_manager()
    state = state_mgr.get_or_create(session_id)
    rewritten = rewrite_query(original_query, state)
    final_query = rewritten

    # P5: 网关语义缓存优先 (Milvus), 回落现有 SQLite 缓存
    cached = None
    try:
        from src.cache.gateway_cache import gateway_cache
        cached = gateway_cache.get_cached(final_query)
    except Exception:
        cached = None
    if cached is None:
        cached = cache_get(final_query)

    # ── 2. 缓存命中 → 直接返回 ──────────────────
    if cached:
        cached["session_id"] = session_id
        if rewritten != original_query:
            cached["rewritten"] = rewritten
        try:
            from src.monitor.metrics_store import metrics_store
            metrics_store.incr_cache_hit()
        except Exception:
            pass
        # 缓存命中的 handoff 回答同样建工单 (闭环一致性)
        if cached.get("handoff"):
            cached["ticket_id"] = _create_handoff_ticket(session_id, user_id, original_query)
        logger.info("语义缓存命中: session=%s, query=%s",
                    session_id[:8], original_query[:40])
        # 后台写日志 (executor, 不阻塞响应)
        loop = asyncio.get_running_loop()
        asyncio.ensure_future(_run_in_executor(
            loop, _do_log_cache_hit,
            session_id, rewritten, state_mgr, cached, original_query
        ))
        return cached

    # ── 3. 规则路由 (同步, <0.1ms) ──────────────
    qclass = route_class(final_query)

    # 只有 greeting 保留内联: 它不查 KB、不需要意图, 走图反而要过 memory_node 更慢。
    # simple_fact 已收进图 —— 检索与五道闸判定统一在 route_retrieve_node,
    # 此前内联那条走 top-5 检索、不做意图分类、还伪造 review 分数, 与流式路径不一致。
    if qclass == "greeting":
        _t0 = time.time()
        try:
            # ☆ ASYNC LLM 调用 — 不占用任何线程 ☆
            answer = await create_client(cheap=True).chat_async([
                {"role": "system", "content": "你是一个友好的客服助手。请用中文简洁回答。\n\n" + BOUNDARY_BLOCK_LIGHT},
                {"role": "user", "content": final_query},
            ])
            result = {
                "answer": answer,
                "session_id": session_id,
                "entities": None,
                "entities_raw": [],
                "_intent": "chitchat",
                "_confidence": None,   # greeting 没有真实置信度, 不编
                "_rule_result": qclass,
                "sources": [],
                # greeting 不经 reviewer: 如实标注 skipped, 分数给 null 而不是写死一个 pass 分数
                # (前端 types/api.ts 把 score 声明为必填的 number|null, 省略会让它变 undefined)
                "review": {"score": None, "verdict": "skipped", "claims": 0},
                "rewritten": rewritten if rewritten != original_query else None,
            }
            _apply_compliance(result)
            # 后台写缓存 + 日志 (executor, 不阻塞响应)
            loop = asyncio.get_running_loop()
            _tid = get_trace_id_snapshot()
            result["trace_id"] = _tid
            asyncio.ensure_future(_run_in_executor(
                loop, partial(_do_post_process, trace_id=_tid, user_id=user_id),
                final_query, result, session_id, state_mgr,
                rewritten, original_query, time.time() - _t0, qclass
            ))
            logger.info("快速通道 greeting (async): session=%s, query=%s",
                        session_id[:8], original_query[:40])
            return result
        except Exception as e:
            logger.warning("greeting 内联失败, 降级到完整Pipeline: %s", e)
            # 继续走完整 Pipeline

    # ── 4. 完整 Pipeline (WorkerPool, 限 2 线程) ──
    _t0 = time.time()
    from src.core.worker_pool import get_worker_pool

    try:
        result = await asyncio.wait_for(
            get_worker_pool().run_analysis_async(
                final_query, timeout=SYNC_TIMEOUT, cache_check=False,
                state=state, user_id=user_id,
            ),
            timeout=SYNC_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("聊天超时: session=%s, query=%s",
                     session_id[:8], original_query[:40])
        try:
            cached = cache_get(final_query)
            if cached:
                cached["session_id"] = session_id
                cached["_degraded"] = True
                return cached
        except Exception as e:
            logger.debug("超时降级缓存读取失败 (可忽略): %s", e)
        raise HTTPException(
            status_code=504,
            detail=f"请求超时 (超过{SYNC_TIMEOUT}秒)，请简化问题后重试",
        )
    except Exception as e:
        logger.error("聊天请求失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

    # 出站合规检查 (掩码敏感词)
    _apply_compliance(result)
    # 转人工工单 (落库 + 回传 ticket_id)
    if result.get("handoff"):
        result["ticket_id"] = _create_handoff_ticket(session_id, user_id, original_query)

    # 后台写缓存 + 日志 (executor, 不阻塞响应)
    loop = asyncio.get_running_loop()
    _tid = get_trace_id_snapshot()
    result["trace_id"] = _tid          # 回传前端, 供反馈关联
    asyncio.ensure_future(_run_in_executor(
        loop, partial(_do_post_process, trace_id=_tid, user_id=user_id),
        final_query, result, session_id, state_mgr,
        rewritten, original_query, time.time() - _t0
    ))

    logger.info("聊天完成: session=%s, query=%s, sources=%d",
                session_id[:8], original_query[:40],
                len(result.get("sources", [])))
    return result


# ── 后台任务辅助函数 ──────────────────

async def _run_in_executor(loop, func, *args):
    """在 executor 中运行同步函数, 异常不传播到事件循环"""
    try:
        await loop.run_in_executor(None, func, *args)
    except Exception as e:
        logger.debug("后台任务失败 (可忽略): %s", e)


def _do_log_cache_hit(session_id, rewritten, state_mgr, cached, original_query):
    """缓存命中日志 (在 executor 线程中运行)"""
    try:
        from src.db.manager import db_manager
        db_manager.log_query(
            query=original_query, latency=0, cache_hit=True,
            intent=cached.get("_intent"),
            confidence=cached.get("_confidence"),
            rule_result=cached.get("_rule_result"),
        )
    except Exception as e:
        logger.debug("缓存命中日志写入失败 (可忽略): %s", e)
    try:
        state_mgr.update(
            session_id=session_id, query=rewritten,
            answer=cached.get("answer", ""),
            entities=cached.get("entities_raw"),
            intent=cached.get("_intent"),
        )
    except Exception as e:
        logger.debug("缓存命中状态更新失败 (可忽略): %s", e)


def _do_post_process(final_query, result, session_id, state_mgr,
                      rewritten, original_query, latency, intent=None,
                     trace_id="", user_id="anonymous", cache_hit=False):
    """后台缓存写入 + 状态更新 + DB 日志 + trace (在 executor 线程中运行)

    trace_id/user_id 必须由调用方从请求上下文传入 — ContextVar 不跨线程。
    """
    # 槽位追问等有状态响应不写缓存 (依赖会话上下文)
    if result.get("_followup"):
        logger.debug("追问响应跳过缓存: %s", final_query[:30])
        return
    try:
        cache_set = None
        from src.core.answer_cache import set_cache
        try:
            set_cache(final_query, result)
        except Exception as e:
            logger.debug("缓存写入失败 (可忽略): %s", e)
        # P5: 网关语义缓存回写
        try:
            from src.cache.gateway_cache import gateway_cache
            gateway_cache.set_cache(final_query, result)
        except Exception as e:
            logger.debug("网关缓存写入失败 (可忽略): %s", e)
    except Exception as e:
        logger.debug("缓存写入导入失败 (可忽略): %s", e)

    try:
        state_mgr.update(
            session_id=session_id, query=rewritten,
            answer=result.get("answer", ""),
            entities=result.get("entities_raw"),
            intent=result.get("_intent"),
        )
    except Exception as e:
        logger.debug("后台状态更新失败 (可忽略): %s", e)

    try:
        from src.db.manager import db_manager
        db_manager.log_query(
            query=original_query, latency=latency,
            tokens=0, cache_hit=False,
            intent=intent or result.get("_intent"),
        )
    except Exception as e:
        logger.debug("后台 DB 日志写入失败 (可忽略): %s", e)

    try:
        from src.db.manager import db_manager
        db_manager.log_trace(
            trace_id=trace_id, session_id=session_id, query=original_query,
            rewritten=rewritten, result=result, latency=latency,
            user_id=user_id, cache_hit=cache_hit,
        )
    except Exception as e:
        logger.debug("后台 trace 写入失败 (可忽略): %s", e)


# ── 流式接口 ──────────────────────────────────

@router.post("/stream")
async def chat_stream(req: ChatRequest, authorization: Optional[str] = Header(None)):
    """SSE 流式聊天接口"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    session_id = req.session_id or str(uuid.uuid4())

    # 解析当前用户 (订单归属校验用)
    user_id = req.user_id
    if not user_id:
        from src.api.routes.auth import resolve_user_id
        user_id = resolve_user_id(authorization) or ""

    # 流式延迟打点: t0 从**进入处理器**算起(含排队与改写), 首个 token 到时记 TTFB。
    # 不能靠中间件测 SSE —— 那里只能测到「响应对象返回」(实测 ~1ms)。
    timing = {"t0": time.time(), "ttfb": None}
    return StreamingResponse(
        _timed_stream(_stream_chat_response(req.query, session_id, user_id=user_id,
                                            timing=timing), timing),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _timed_stream(gen, timing: dict):
    """给 SSE 流打点: 流结束时记首 token 时间(TTFB)与总时长、成败。

    客户端中断也会走到 finally, 此时记 ok=False —— 中断是真实结果, 不该被算成成功。
    指标是旁路: 记录失败只记 debug, 异常照常向上抛(不吞)。
    """
    ok = False
    try:
        async for chunk in gen:
            yield chunk
        ok = True
    finally:
        ttfb = timing.get("ttfb")
        total = round((time.time() - timing.get("t0", time.time())) * 1000)
        # 同一组数字既进指标库(在线看板)也进日志(离线复盘) —— 两条路可对照
        logger.info(
            "POST /api/v1/chat/stream -> %d (ttfb %s / total %dms)",
            200 if ok else 500, f"{ttfb}ms" if ttfb is not None else "-", total,
            extra={"method": "POST", "path": "/api/v1/chat/stream",
                   "status": 200 if ok else 500, "stream": True,
                   "latency_ms": total, "ttfb_ms": ttfb},
        )
        try:
            from src.monitor.metrics_store import metrics_store
            metrics_store.record_stream(ttfb_ms=ttfb, total_ms=total, ok=ok)
        except Exception as e:
            logger.debug("流式指标记录失败 (可忽略): %s", e)


async def _stream_chat_response(query: str, session_id: str, user_id: str = "",
                                timing: Optional[dict] = None):
    """流式生成聊天响应"""

    def _event(event_type: str, **kwargs) -> str:
        # 首个 token(或缓存命中直接出 result)到达时记 TTFB —— 这才是用户感知的首字时间;
        # 前面的 stage/info 帧是秒回的, 拿它们当 TTFB 就又变成了假象。
        if timing is not None and timing.get("ttfb") is None \
                and event_type in ("token", "result"):
            timing["ttfb"] = round((time.time() - timing["t0"]) * 1000)
        return f"data: {json.dumps({'type': event_type, **kwargs}, ensure_ascii=False)}\n\n"

    loop = asyncio.get_running_loop()

    def _init():
        from src.core.dialog_state import get_state_manager
        from src.llm.query_rewriter import rewrite_query
        state_mgr = get_state_manager()
        state = state_mgr.get_or_create(session_id)
        rewritten = rewrite_query(query.strip(), state)
        from src.core.answer_cache import get_cached as cache_get
        # P5: 网关语义缓存优先 (Milvus), 回落现有 SQLite 缓存
        cached = None
        try:
            from src.cache.gateway_cache import gateway_cache
            cached = gateway_cache.get_cached(rewritten)
        except Exception:
            cached = None
        if cached is None:
            cached = cache_get(rewritten)
        return state_mgr, state, rewritten, cached, cache_get

    init_result = await loop.run_in_executor(None, _init)
    state_mgr, state, rewritten, cached, cache_get_func = init_result

    yield _event("stage", stage="rewriting", message="🔄 正在理解对话上下文...")
    if rewritten != query:
        yield _event("info", message=f"查询补全: 「{query}」→ 「{rewritten}」")

    if cached:
        cached["session_id"] = session_id
        if rewritten != query:
            cached["rewritten"] = rewritten
        try:
            from src.monitor.metrics_store import metrics_store
            metrics_store.incr_cache_hit()
        except Exception:
            pass
        # 缓存命中的 handoff 回答同样建工单 + 推事件 (闭环一致性)
        if cached.get("handoff"):
            tid = _create_handoff_ticket(session_id, user_id, query)
            yield _event("handoff", message="正在为您转接人工客服，请稍候…", ticket_id=tid)
        yield _event("result", **{k: v for k, v in cached.items()
                                  if k in ("answer", "sources", "entities",
                                           "review", "session_id", "rewritten")})
        yield _event("done")
        logger.info("流式缓存命中: %s", query[:30])
        return

    from src.core.fast_path import route_class
    qclass = route_class(query)
    if qclass == "greeting":
        from src.llm.llm_client import create_client
        # 寒暄同样逐 token 流式推送 (首 token 也由 _event 计入 TTFB)
        _gparts = []
        async for _tok in create_client(cheap=True).chat_stream_async([
            {"role": "system", "content": "你是一个友好的客服助手。请用中文简洁回答。"},
            {"role": "user", "content": query},
        ]):
            _gparts.append(_tok)
            yield _event("token", token=_tok)
        ans = "".join(_gparts)
        yield _event("result", answer=ans, sources=[], entities=None,
                     review={"score": None, "verdict": "skipped", "claims": 0})
        yield _event("done")
        return

    token_queue = Queue()
    stop_event = threading.Event()
    _stream_start = time.time()

    def _run_analysis():
        from src.graph.cs_graph import run_graph_stream
        from src.business.context import set_current_user
        from src.core.concurrency import processing_slot
        set_current_user(user_id or None)  # 空则清除, 避免线程间泄漏
        try:
            # P6: 门控 LLM 密集处理并发 (超时 → 降级, 不拒绝)
            with processing_slot():
                # P3: LangGraph 多 Agent 状态机 (节点事件 + token 流式)
                run_graph_stream(token_queue, rewritten,
                                 session_id=session_id, user_id=user_id,
                                 slots=dict(state.slots) if state else {})
        except TimeoutError as e:
            logger.warning("处理并发门控超时: %s", str(e)[:80])
            token_queue.put({"type": "error", "message": str(e)[:200]})
        except Exception as e:
            logger.error("分析线程异常: %s", e, exc_info=True)
            token_queue.put({"type": "error",
                             "message": f"分析引擎异常: {str(e)[:200]}"})
        finally:
            stop_event.set()

    thread = threading.Thread(target=_run_analysis, daemon=True)
    thread.start()

    try:
        while True:
            if time.time() - _stream_start > STREAM_TIMEOUT:
                yield _event("error", message="请求超时，请稍后重试")
                break
            if stop_event.is_set() and token_queue.empty():
                break
            try:
                msg = token_queue.get_nowait()
            except QueueEmpty:
                await asyncio.sleep(0.05)
                continue

            msg_type = msg.get("type")

            if msg_type == "stage":
                yield _event("stage", **{k: v for k, v in msg.items()
                                        if k != "type"})
            elif msg_type == "token":
                yield _event("token", token=msg["token"])
            elif msg_type == "result_meta":
                yield _event("result_meta", **{k: v for k, v in msg.items()
                                              if k != "type"})
            elif msg_type == "result":
                _tid = get_trace_id_snapshot()
                msg["trace_id"] = _tid
                _apply_compliance(msg)
                # P3: 图算出的会话槽位写回 dialog_state
                if msg.get("slots") and state is not None:
                    try:
                        state.slots.update(msg["slots"])
                    except Exception as e:
                        logger.debug("槽位回写失败 (可忽略): %s", e)
                if msg.get("handoff"):
                    tid = _create_handoff_ticket(session_id, user_id, query)
                    yield _event("handoff", message="正在为您转接人工客服，请稍候…", ticket_id=tid)
                yield _event("result", **{k: v for k, v in msg.items()
                                         if k not in ("type", "slots", "_confidence")})
                msg["session_id"] = session_id
                if rewritten != query:
                    msg["rewritten"] = rewritten
                # P5: 流式路径缓存回写 (后台 executor, 不阻塞 SSE)
                try:
                    loop3 = asyncio.get_running_loop()
                    asyncio.ensure_future(_run_in_executor(
                        loop3, partial(_do_post_process, trace_id=_tid, user_id=user_id),
                        rewritten, msg, session_id, state_mgr,
                        rewritten, query, time.time() - _stream_start, msg.get("intent")
                    ))
                except Exception as e:
                    logger.debug("流式缓存回写调度失败 (可忽略): %s", e)
            elif msg_type == "error":
                yield _event("error", message=msg.get("message", "未知错误"))
            elif msg_type == "done":
                yield _event("done")
                break
    finally:
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=2)
        yield _event("done")


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """收集用户对回答的反馈 (rating: 1 赞 / -1 踩), 用于优化与转人工"""
    if req.rating not in (1, -1, 0):
        raise HTTPException(status_code=400, detail="rating 必须为 1 / -1 / 0")
    from src.db.manager import db_manager
    try:
        fid = db_manager.add_feedback(req.session_id, req.rating, req.comment,
                                      req.user_id, trace_id=req.trace_id)
        return {"ok": True, "feedback_id": fid}
    except Exception as e:
        logger.error("反馈记录失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="反馈记录失败")
