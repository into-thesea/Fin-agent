"""
Fin-Agent WebSocket 实时任务状态推送

架构:
  Celery Worker → Redis Pub/Sub → WebSocket Server → 浏览器

流程:
  1. ETL 任务 (celery) 每步写入 Redis hash + 发布到 Redis channel
  2. WebSocket 服务器订阅 Redis channel
  3. 消息通过 WebSocket 推送到前端

启动 (与 FastAPI 集成):
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000

单独启动 WebSocket 服务器:
  python src/ws_manager.py
"""

import os
import json
import asyncio
import logging
from typing import Set, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 确保 .env 加载 (避免 REDIS_HOST 默认成 localhost → IPv6 超时)
from dotenv import load_dotenv
load_dotenv()

try:
    import redis.asyncio as aioredis
    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False

try:
    from fastapi import WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Pub/Sub Channel
ETL_CHANNEL = "fin:etl:progress"


class ConnectionManager:
    """
    WebSocket 连接管理器

    维护所有活跃连接，支持按 task_id 订阅/广播。
    """

    def __init__(self):
        # {task_id: set(WebSocket)}
        self._connections: dict[str, Set[WebSocket]] = {}
        # {websocket: set(task_ids)}
        self._subscriptions: dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        """接受 WebSocket 连接并订阅任务"""
        await websocket.accept()
        if task_id not in self._connections:
            self._connections[task_id] = set()
        self._connections[task_id].add(websocket)
        if websocket not in self._subscriptions:
            self._subscriptions[websocket] = set()
        self._subscriptions[websocket].add(task_id)
        logger.info("WebSocket 已连接: task=%s, clients=%d", task_id, self._connections[task_id])

    async def disconnect(self, websocket: WebSocket, task_id: str = None):
        """断开 WebSocket 连接"""
        if task_id and task_id in self._connections:
            self._connections[task_id].discard(websocket)
            if not self._connections[task_id]:
                del self._connections[task_id]

        if websocket in self._subscriptions:
            if task_id:
                self._subscriptions[websocket].discard(task_id)
            if not self._subscriptions[websocket]:
                del self._subscriptions[websocket]

    async def broadcast_task(self, task_id: str, message: dict):
        """向订阅了某任务的所有客户端广播"""
        if task_id not in self._connections:
            return
        disconnected = set()
        payload = json.dumps(message, ensure_ascii=False)
        for ws in self._connections[task_id]:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            await self.disconnect(ws, task_id)

    async def broadcast_all(self, message: dict):
        """向所有连接广播 (用于系统公告)"""
        payload = json.dumps(message, ensure_ascii=False)
        for task_id in list(self._connections.keys()):
            await self.broadcast_task(task_id, message)

    @property
    def active_connections(self) -> int:
        return sum(len(ws_set) for ws_set in self._connections.values())

    @property
    def active_tasks(self) -> int:
        return len(self._connections)


# 全局连接管理器
manager = ConnectionManager()


# ──────────────────────────────────────────────
# Redis Pub/Sub 监听器
# ──────────────────────────────────────────────

class RedisProgressListener:
    """
    Redis Pub/Sub 监听器

    在后台订阅 ETL 进度 channel，将消息广播到对应的 WebSocket 客户端。
    需要在 FastAPI 应用启动时作为 background task 运行。
    """

    def __init__(self):
        self._redis = None
        self._pubsub = None
        self._running = False

    async def start(self):
        """启动 Redis 订阅监听"""
        if not ASYNC_REDIS_AVAILABLE:
            logger.warning("redis.asyncio 不可用，WebSocket 进度推送降级")
            return

        try:
            self._redis = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=5,
            )
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(ETL_CHANNEL)
            self._running = True
            logger.info("Redis Pub/Sub 监听器已启动: channel=%s", ETL_CHANNEL)
        except Exception as e:
            logger.error("Redis Pub/Sub 启动失败: %s", e)
            self._running = False

    async def listen(self):
        """监听循环 (在后台任务中运行)"""
        if not self._pubsub:
            return

        while self._running:
            try:
                message = await self._pubsub.get_message(
                    timeout=1.0, ignore_subscribe_messages=True
                )
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    task_id = data.get("task_id")
                    if task_id:
                        await manager.broadcast_task(task_id, data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Pub/Sub 消息处理出错: %s", e)
                await asyncio.sleep(1)

    async def stop(self):
        """停止监听"""
        self._running = False
        if self._pubsub:
            await self._pubsub.unsubscribe(ETL_CHANNEL)
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        logger.info("Redis Pub/Sub 监听器已停止")


listener = RedisProgressListener()


# ──────────────────────────────────────────────
# Celery 任务进度发布 (同步 Redis 客户端)
# ──────────────────────────────────────────────

def publish_progress(task_id: str, status: str, progress: int,
                     stage: str = "", error: str = ""):
    """
    发布 ETL 进度到 Redis Pub/Sub

    被 Celery 任务调用 (同步上下文):
      from src.core.ws_manager import publish_progress
      publish_progress(task_id, "processing", 50, "vectorizing")
    """
    try:
        import redis as sync_redis
        r = sync_redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD or None,
            decode_responses=True,
        )
        message = {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if error:
            message["error"] = error

        r.publish(ETL_CHANNEL, json.dumps(message, ensure_ascii=False))
        r.close()
    except Exception as e:
        logger.debug("Pub/Sub 发布失败: %s", e)


# ──────────────────────────────────────────────
# FastAPI WebSocket 端点
# ──────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

    ws_router = APIRouter()

    @ws_router.websocket("/ws/task/{task_id}")
    async def task_websocket_endpoint(websocket: WebSocket, task_id: str):
        """
        WebSocket 端点: /ws/task/{task_id}

        客户端连接后实时接收 ETL 任务进度。
        """
        await manager.connect(websocket, task_id)
        try:
            # 保持连接，等待消息 (或 ping/pong)
            while True:
                try:
                    data = await websocket.receive_text()
                    # 客户端可以发送 ping，服务端回复 pong
                    if data == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except WebSocketDisconnect:
                    break
        finally:
            await manager.disconnect(websocket, task_id)
else:
    ws_router = None


# ──────────────────────────────────────────────
# 独立启动 WebSocket 测试服务器
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # 测试模式：启动一个简单的 WebSocket echo server
    import asyncio

    async def main():
        print("WebSocket Manager 测试模式")
        print(f"  Redis: {REDIS_HOST}:{REDIS_PORT}")
        print(f"  Channel: {ETL_CHANNEL}")
        print(f"  Async Redis: {'可用' if ASYNC_REDIS_AVAILABLE else '不可用'}")
        print(f"  FastAPI WebSocket: {'可用' if FASTAPI_AVAILABLE else '不可用'}")

        # 测试发布一条进度消息
        publish_progress("test_task_001", "processing", 50, "测试阶段")
        print("  测试消息已发布")

    asyncio.run(main())
