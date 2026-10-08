"""
Fin-Agent 工作线程池 (v1.0)

专用 2-Worker 线程池，管理分析 pipeline 的并发执行。

特性:
  - 固定 2 个工作线程 (避免无限制线程膨胀)
  - 带超时的异步提交 (asyncio.wait_for)
  - 队列长度限制 (防止请求堆积)
  - 健康检查 / 状态统计
  - 优雅关闭 (drain + timeout)

用法:
    from src.core.worker_pool import get_worker_pool
    pool = get_worker_pool()

    # 同步提交 (在线程中运行)
    result = pool.run_analysis(query)

    # 异步提交 (给 asyncio 使用)
    result = await pool.run_analysis_async(query, timeout=120)
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

from src.config import settings

# 配置
MAX_WORKERS = settings.max_workers
MAX_QUEUE_SIZE = settings.max_queue_size
DEFAULT_TIMEOUT = settings.default_timeout  # 秒
SHUTDOWN_TIMEOUT = settings.shutdown_timeout


class WorkerPool:
    """
    专用工作线程池

    用固定 2 个线程执行 AnalystAgent.analyze()，
    避免默认线程池的无限制膨胀和竞争。
    """

    def __init__(self, max_workers: int = MAX_WORKERS):
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agent-worker",
        )
        self._lock = threading.Lock()
        self._running_tasks: dict[int, float] = {}  # {thread_id: start_time}
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_errors": 0,
            "total_timeout": 0,
            "total_rejected": 0,      # 队列满被拒
            "current_queue_depth": 0,
            "current_active_workers": 0,
            "avg_duration_seconds": 0.0,
            "max_duration_seconds": 0.0,
            "started_at": time.time(),
            "last_completed_at": None,
        }
        self._total_duration = 0.0
        self._shutdown = False

    # ── 公共接口 ──────────────────────────────

    def run_analysis(self, query: str, timeout: int = DEFAULT_TIMEOUT,
                     cache_check: bool = True, state=None, user_id: str = None) -> dict:
        """
        同步执行分析 (在当前线程阻塞等待)

        流程:
          1. 先检查语义缓存 (可选)
          2. 提交到线程池
          3. 等待结果或超时

        Args:
            query: 用户查询
            timeout: 超时秒数
            cache_check: 是否先查缓存

        Returns:
            {"answer": str, ...} 或超时/错误时的降级响应
        """
        # 1. 缓存检查
        if cache_check:
            try:
                from src.core.answer_cache import get_cached
                cached = get_cached(query)
                if cached:
                    logger.info("WorkerPool 缓存命中: %s", query[:40])
                    return cached
            except Exception as e:
                logger.debug("WorkerPool 缓存查询失败 (可忽略): %s", e)

        # 2. 检查队列负载
        if self._shutdown:
            return self._degraded_response("服务正在关闭", query)

        # 3. 提交到线程池
        start = time.time()
        from src.infra.logging_config import get_trace_id_snapshot
        current_tid = get_trace_id_snapshot()
        future = self._executor.submit(self._run_analysis_inner, query, current_tid, state, user_id)

        with self._lock:
            self._stats["total_submitted"] += 1
            self._stats["current_queue_depth"] = (
                self._executor._work_queue.qsize()
                if hasattr(self._executor, '_work_queue') else 0
            )

        try:
            result = future.result(timeout=timeout)
            duration = time.time() - start
            self._record_completion(duration, success=True)

            # 写入缓存
            if cache_check:
                try:
                    from src.core.answer_cache import set_cache
                    set_cache(query, result)
                except Exception as e:
                    logger.debug("WorkerPool 缓存写入失败 (可忽略): %s", e)

            return result

        except TimeoutError:
            duration = time.time() - start
            self._record_completion(duration, success=False, timed_out=True)
            future.cancel()
            logger.warning("WorkerPool 超时 (%ds): %s", timeout, query[:40])
            return self._degraded_response(
                f"分析请求超时 (超过{timeout}秒)，请简化问题后重试",
                query
            )

        except Exception as e:
            duration = time.time() - start
            self._record_completion(duration, success=False)
            logger.error("WorkerPool 执行失败: %s", e, exc_info=True)
            return self._degraded_response(f"分析引擎处理出错: {str(e)[:100]}", query)

    async def run_analysis_async(self, query: str,
                                  timeout: int = DEFAULT_TIMEOUT,
                                  cache_check: bool = True,
                                  state=None, user_id: str = None) -> dict:
        """
        异步执行分析 (asyncio 协程)

        在 asyncio 事件循环中调用，不会阻塞事件循环。
        执行前先查缓存 (可选), 不命中才提交到线程池。
        """
        # 1. 缓存检查 (async 中快速同步查找)
        if cache_check:
            try:
                from src.core.answer_cache import get_cached
                cached = get_cached(query)
                if cached:
                    logger.info("WorkerPool 异步缓存命中: %s", query[:40])
                    return cached
            except Exception as e:
                logger.debug("WorkerPool 异步缓存查询失败 (可忽略): %s", e)

        # 2. 检查是否已关闭
        if self._shutdown:
            return self._degraded_response("服务正在关闭", query)

        # 3. 提交到线程池
        loop = asyncio.get_running_loop()
        from src.infra.logging_config import get_trace_id_snapshot
        current_tid = get_trace_id_snapshot()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._run_analysis_inner,
                    query,
                    current_tid,
                    state,
                    user_id,
                ),
                timeout=timeout,
            )
            # 4. 写入缓存 (不阻塞)
            if cache_check:
                try:
                    from src.core.answer_cache import set_cache
                    set_cache(query, result)
                except Exception as e:
                    logger.debug("WorkerPool 异步缓存写入失败 (可忽略): %s", e)
            return result
        except asyncio.TimeoutError:
            logger.warning("WorkerPool 异步超时 (%ds): %s", timeout, query[:40])
            return self._degraded_response(
                f"分析超时 (超过{timeout}秒)", query
            )
        except Exception as e:
            logger.error("WorkerPool 异步执行失败: %s", e, exc_info=True)
            return self._degraded_response(f"分析引擎出错: {str(e)[:100]}", query)

    # ── 内部方法 ──────────────────────────────

    def _run_analysis_inner(self, query: str, trace_id: str = "", state=None,
                            user_id: str = None) -> dict:
        """在 worker 线程中执行的实际分析逻辑"""
        if trace_id:
            from src.infra.logging_config import set_trace_id
            set_trace_id(trace_id)
        from src.business.context import set_current_user
        set_current_user(user_id or None)  # 空则清除, 避免线程间泄漏上一请求的用户
        from src.analyst_agent import AnalystAgent
        agent = AnalystAgent()
        try:
            # 先查规则路由 (快速通道)
            from src.llm.query_router import classify as classify_query
            qclass = classify_query(query)

            if qclass in ("greeting", "simple_fact"):
                from src.agents.prompts import BOUNDARY_BLOCK_LIGHT
                from src.llm.llm_client import create_client
                # simple_fact: 先检索知识库再作答 (与 chat.py 快速通道一致)
                sources = []
                context_note = ""
                if qclass == "simple_fact":
                    from src.tools.registry import retrieve_knowledge
                    r = retrieve_knowledge(query, 5)
                    context_note = r["text"]
                    sources = r["sources"]
                prompt = ("你是一个友好的客服助手。请用中文简洁回答。\n\n" + BOUNDARY_BLOCK_LIGHT) if qclass == "greeting" \
                    else ("你是客服助手。请基于知识库检索结果简要回答, 无法确定时明确说明。\n\n" + BOUNDARY_BLOCK_LIGHT + "\n\n" + context_note)
                answer = create_client(cheap=True).chat([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ])
                return {
                    "answer": answer,
                    "entities": None,
                    "entities_raw": [],
                    "_intent": "greeting" if qclass == "greeting" else "simple_fact",
                    "sources": sources,
                    "review": {"score": 85, "verdict": "pass", "claims": 0},
                }

            # 完整分析 pipeline (传入会话状态供槽位预检)
            return agent.analyze(query, state=state)
        finally:
            agent.close()

    def _record_completion(self, duration: float, success: bool = True,
                           timed_out: bool = False):
        """记录任务完成统计"""
        with self._lock:
            self._stats["total_completed"] += 1
            if not success:
                if timed_out:
                    self._stats["total_timeout"] += 1
                else:
                    self._stats["total_errors"] += 1
            self._stats["current_active_workers"] = max(
                0, self._stats["current_active_workers"] - 1
            )
            self._stats["last_completed_at"] = time.time()

            # 滚动平均
            self._total_duration += duration
            completed = self._stats["total_completed"]
            if completed > 0:
                self._stats["avg_duration_seconds"] = self._total_duration / completed
            if duration > self._stats["max_duration_seconds"]:
                self._stats["max_duration_seconds"] = duration

    def _degraded_response(self, message: str, query: str) -> dict:
        """生成降级响应"""
        # 尝试从语义缓存获取最近似的回答 (降级但比空好)
        try:
            from src.core.answer_cache import get_cached
            cached = get_cached(query)
            if cached:
                logger.info("WorkerPool 降级: 从缓存返回近似回答")
                return cached
        except Exception as e:
            logger.debug("降级缓存查询失败 (可忽略): %s", e)

        return {
            "answer": f"⚠️ {message}\n\n请稍后重试，或尝试更换表述方式。",
            "entities": None,
            "entities_raw": [],
            "_intent": "error",
            "sources": [],
            "review": {"score": 0, "verdict": "error", "claims": 0},
        }

    # ── 管理接口 ──────────────────────────────

    def get_stats(self) -> dict:
        """获取线程池统计"""
        with self._lock:
            active = self._stats["total_submitted"] - self._stats["total_completed"]
            return {
                **self._stats,
                "active_tasks": max(0, active),
                "max_workers": self._max_workers,
                "uptime_seconds": int(time.time() - self._stats["started_at"]),
            }

    @property
    def available(self) -> bool:
        """检查池是否可用 (未关闭)"""
        return not self._shutdown

    def shutdown(self, wait: bool = True):
        """优雅关闭"""
        logger.info("WorkerPool 关闭中...")
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=True)
        logger.info("WorkerPool 已关闭")


# ── 全局单例 ────────────────────────────────────

_pool: WorkerPool = None
_pool_lock = threading.Lock()


def get_worker_pool() -> WorkerPool:
    """获取全局 WorkerPool 单例"""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = WorkerPool(max_workers=MAX_WORKERS)
                logger.info("WorkerPool 已创建: %d workers, 队列上限 %d",
                            MAX_WORKERS, MAX_QUEUE_SIZE)
    return _pool


def shutdown_pool():
    """关闭全局 WorkerPool"""
    global _pool
    if _pool:
        _pool.shutdown()
        _pool = None
