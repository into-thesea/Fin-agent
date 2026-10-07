"""
Fin-Agent 企业级缓存层

多级缓存架构:
  L1: 客户端缓存 (React Query, staleTime=5min)
  L2: Redis 缓存 (当前实现)
  L3: 数据层 (Neo4j / PostgreSQL)

缓存类型:
  1. 向量检索缓存 — 相同 query embedding 直接返回 TOP-K 结果
  2. 图谱查询缓存 — 相同实体邻接查询直接返回
  3. 语义缓存 — 相似问题 (cosine > 0.95) 复用 LLM 回答
  4. 任务状态缓存 — ETL 异步任务进度实时查询
"""

import os
import json
import hashlib
import time
import socket
import logging
from typing import Optional, Any

from src.infra.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis 库未安装，缓存层将以降级模式运行 (no-op)")


def semantic_cache_key(query: str) -> str:
    """生成语义缓存的主键 (基于 MD5)"""
    return hashlib.md5(query.encode("utf-8")).hexdigest()


class RedisCache:
    """Redis 缓存客户端 — 封装所有缓存操作"""

    def __init__(self):
        self._client = None
        self._enabled = False
        self._cb = CircuitBreaker("redis", failure_threshold=3, recovery_timeout=30.0)
        self._init_client()

    @staticmethod
    def _check_available() -> bool:
        """Socket 级快速检测 Redis 是否可达（2s 超时，避免 Windows 50s 阻塞 bug）"""
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
            logger.debug("Redis socket 检测失败 (%s:%s → %d), 跳过 Redis 连接", host, port, result)
            return False
        except Exception as e:
            logger.debug("Redis socket 检测异常 (%s:%s → %s), 跳过 Redis 连接", host, port, e)
            return False

    def _do_connect(self, redis_host: str, redis_port: int):
        """实际的 Redis 连接逻辑（由断路器保护）"""
        if not self._check_available():
            raise ConnectionError(f"Redis 不可达 ({redis_host}:{redis_port})")

        redis_password = os.getenv("REDIS_PASSWORD", None)
        redis_db = int(os.getenv("REDIS_DB", 0))

        self._client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password or None,
            db=redis_db,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        # 惰性验证: 不在这里 ping() (WSL Redis ping 耗时 3s, 阻塞事件循环)
        # 首次实际读写时自动建立连接, 失败由 _disable() 处理
        logger.info("Redis 缓存已连接: %s:%s (db=%d)", redis_host, redis_port, redis_db)

    def _init_client(self):
        if not REDIS_AVAILABLE:
            logger.info("Redis 库未安装，缓存层运行在 no-op 模式")
            return

        # 断路器检查: OPEN 状态直接跳过连接
        if self._cb.state == CircuitState.OPEN:
            self._enabled = False
            logger.debug("Redis 断路器已熔断，跳过连接")
            return

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        try:
            self._cb.call(self._do_connect, redis_host, redis_port)
            self._enabled = True
        except CircuitBreakerOpenError:
            self._enabled = False
            logger.warning("Redis 断路器熔断，缓存降级为 no-op (%s:%s)", redis_host, redis_port)
        except Exception as e:
            self._enabled = False
            logger.warning("Redis 连接失败 (%s:%s), 缓存降级为 no-op: %s", redis_host, redis_port, e)

    def _disable(self):
        """遇到连接错误时降级缓存，并通知断路器"""
        self._enabled = False
        self._client = None
        self._cb._on_failure()

    def close(self):
        if self._client:
            self._client.close()

    @property
    def enabled(self) -> bool:
        """检查缓存是否启用，当断路器允许时尝试自动恢复"""
        if not self._enabled and self._cb.state != CircuitState.OPEN:
            logger.debug("Redis 缓存降级中，但断路器允许恢复，尝试重新连接")
            self._init_client()
        return self._enabled

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """暴露断路器实例，供监控使用"""
        return self._cb

    # ──────────────────────────────────────────────
    # 1. 向量检索缓存
    # ──────────────────────────────────────────────

    def _vector_cache_key(self, query_hash: str, top_k: int) -> str:
        return f"fin:vector:q:{query_hash}:k{top_k}"

    def get_vector_results(self, query_embedding_hash: str, top_k: int = 5) -> Optional[list]:
        """获取缓存的向量检索结果"""
        if not self._enabled or not self._client:
            return None
        key = self._vector_cache_key(query_embedding_hash, top_k)
        try:
            cached = self._client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            self._disable()
            logger.warning(f"向量缓存读取失败，缓存降级: {e}")
            return None

    def set_vector_results(self, query_embedding_hash: str, top_k: int, results: list, ttl: int = 300):
        """缓存向量检索结果 (默认 5 分钟)"""
        if not self._enabled or not self._client:
            return
        key = self._vector_cache_key(query_embedding_hash, top_k)
        try:
            self._client.setex(key, ttl, json.dumps(results, ensure_ascii=False))
        except Exception as e:
            self._disable()
            logger.warning(f"向量缓存写入失败，缓存降级: {e}")

    # ──────────────────────────────────────────────
    # 2. 图谱查询缓存
    # ──────────────────────────────────────────────

    def _graph_cache_key(self, entity_name: str, relationship_type: str = "") -> str:
        base = f"fin:graph:entity:{entity_name}"
        return f"{base}:rel:{relationship_type}" if relationship_type else base

    def get_graph_relations(self, entity_name: str) -> Optional[list]:
        """获取缓存的实体邻接关系"""
        if not self._enabled or not self._client:
            return None
        key = self._graph_cache_key(entity_name)
        try:
            cached = self._client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            self._disable()
            logger.warning("图谱缓存读取失败, 缓存降级: %s", e)
            return None

    def set_graph_relations(self, entity_name: str, relations: list, ttl: int = 600):
        """缓存图谱查询 (默认 10 分钟)"""
        if not self._enabled or not self._client:
            return
        key = self._graph_cache_key(entity_name)
        try:
            self._client.setex(key, ttl, json.dumps(relations, ensure_ascii=False))
        except Exception as e:
            self._disable()
            logger.debug(f"图谱缓存写入失败: {e}")

    # ──────────────────────────────────────────────
    # 3. 语义缓存 (LLM 响应复用)
    # ──────────────────────────────────────────────

    def get_semantic_response(self, query: str, threshold: float = 0.95) -> Optional[dict]:
        """
        语义缓存查询:
        如果新 query 与某条已缓存 query 的 embedding 余弦相似度 > threshold
        则直接返回缓存的 LLM 回答。

        Returns: {"query": str, "response": str, "similarity": float} 或 None
        """
        if not self._enabled:
            return None
        try:
            all_keys = self._client.scan_iter(match="fin:semantic:*", count=200)
            for key in all_keys:
                cached_data = self._client.hgetall(key)
                if not cached_data:
                    continue
                similarity = self._cosine_sim_float(
                    query, cached_data.get("query", "")
                )
                if similarity > threshold:
                    return {
                        "query": cached_data["query"],
                        "response": cached_data["response"],
                        "similarity": round(similarity, 4),
                    }
        except Exception as e:
            logger.debug(f"语义缓存查询失败: {e}")
        return None

    def set_semantic_response(self, query: str, response: str, ttl: int = 3600):
        """缓存 LLM 回答 (默认 1 小时)"""
        if not self._enabled:
            return
        key = f"fin:semantic:{semantic_cache_key(query)}"
        try:
            self._client.hset(key, mapping={
                "query": query,
                "response": response,
                "cached_at": time.time(),
            })
            self._client.expire(key, ttl)
        except Exception as e:
            logger.debug(f"语义缓存写入失败: {e}")

    def _cosine_sim_float(self, s1: str, s2: str) -> float:
        """
        轻量级文本相似度 — 基于字符级 n-gram 的 Dice 系数。
        企业级可用 embedding 模型算 cosine，但在此保持无额外依赖。
        """
        if not s1 or not s2:
            return 0.0
        # bigram 交集 / 并集
        bigrams1 = set(s1[i:i+2] for i in range(len(s1)-1))
        bigrams2 = set(s2[i:i+2] for i in range(len(s2)-1))
        if not bigrams1 or not bigrams2:
            return 0.0
        intersection = bigrams1 & bigrams2
        return 2.0 * len(intersection) / (len(bigrams1) + len(bigrams2))

    # ──────────────────────────────────────────────
    # 4. ETL 任务状态缓存 (供 WebSocket 轮询)
    # ──────────────────────────────────────────────

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取异步 ETL 任务的状态"""
        if not self._enabled:
            return None
        key = f"fin:task:{task_id}"
        try:
            data = self._client.hgetall(key)
            if not data:
                return None
            return {
                "task_id": task_id,
                "status": data.get("status", "unknown"),
                "progress": int(data.get("progress", 0)),
                "stage": data.get("stage", ""),
                "error": data.get("error", ""),
                "started_at": data.get("started_at", ""),
            }
        except Exception as e:
            logger.debug("任务状态缓存读取失败: %s", e)
            return None

    def set_task_status(self, task_id: str, status: str, progress: int,
                        stage: str = "", error: str = "", ttl: int = 3600):
        """更新 ETL 任务状态"""
        if not self._enabled:
            return
        key = f"fin:task:{task_id}"
        try:
            self._client.hset(key, mapping={
                "status": status,
                "progress": progress,
                "stage": stage,
                "error": error,
                "updated_at": time.time(),
            })
            self._client.expire(key, ttl)
        except Exception as e:
            logger.debug(f"任务状态写入失败: {e}")

    # ──────────────────────────────────────────────
    # 5. 对话状态持久化 (DST)
    # ──────────────────────────────────────────────

    def get_dialog_state(self, session_id: str) -> Optional[dict]:
        """从 Redis 读取对话状态"""
        if not self._enabled:
            return None
        key = f"fin:dialog:{session_id}"
        try:
            data = self._client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug("对话状态读取失败: %s", e)
        return None

    def set_dialog_state(self, session_id: str, data: dict, ttl: int = 1800):
        """写入对话状态到 Redis (默认 30 分钟 TTL)"""
        if not self._enabled:
            return
        key = f"fin:dialog:{session_id}"
        try:
            self._client.setex(key, ttl, json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.debug("对话状态写入失败: %s", e)

    # ──────────────────────────────────────────────
    # 6. 缓存失效
    # ──────────────────────────────────────────────

    def invalidate_entity_cache(self, entity_name: str):
        """当实体图谱更新时，清除该实体的所有缓存"""
        if not self._enabled:
            return
        try:
            pattern = f"fin:graph:entity:{entity_name}*"
            for key in self._client.scan_iter(match=pattern):
                self._client.delete(key)
        except Exception as e:
            logger.debug(f"缓存失效失败: {e}")

    def invalidate_document_cache(self, document_id: str):
        """当文档更新时，清除该文档相关的向量和语义缓存"""
        if not self._enabled:
            return
        try:
            # 清除该文档的向量缓存 (通过 document_id 模式)
            for key in self._client.scan_iter(match="fin:vector:*"):
                self._client.delete(key)
            # 语义缓存全量失效 (新知识可能影响所有问题的回答)
            for key in self._client.scan_iter(match="fin:semantic:*"):
                self._client.delete(key)
            logger.info(f"文档 {document_id} 相关缓存已失效")
        except Exception as e:
            logger.debug(f"文档缓存失效失败: {e}")

    def invalidate_all(self):
        """全量缓存失效 (极端操作，用于知识库重建后)"""
        if not self._enabled:
            return
        try:
            count = 0
            for key in self._client.scan_iter(match="fin:*"):
                self._client.delete(key)
                count += 1
            logger.info(f"全量缓存已清除: {count} keys")
        except Exception as e:
            logger.debug(f"全量缓存失效失败: {e}")

    # ──────────────────────────────────────────────
    # 6. 监控 & 健康检查
    # ──────────────────────────────────────────────

    @property
    def client(self):
        """原始 redis 客户端 (供指标聚合等直接使用; Redis 不可用时为 None)"""
        return self._client

    def ping(self) -> bool:
        """健康检查 — 使用断路器保护，自动恢复"""
        if not self._client:
            # 断路器允许时尝试重连
            if self._cb.state != CircuitState.OPEN:
                logger.info("Redis 客户端为空但断路器允许恢复，尝试重连")
                self._init_client()
            return self._client is not None

        try:
            self._cb.call(self._client.ping)
            self._enabled = True
            return True
        except CircuitBreakerOpenError:
            self._enabled = False
            logger.warning("Redis ping 被断路器阻止")
            return False
        except Exception as e:
            logger.debug("Redis ping 失败: %s", e)
            self._enabled = False
            return False

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        if not self.enabled or not self._client:
            return {"enabled": False, "circuit_breaker_state": self._cb.state.value}
        try:
            if not self.ping():
                self._disable()
                return {"enabled": False, "circuit_breaker_state": self._cb.state.value}
            info = self._client.info()
            return {
                "enabled": True,
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "total_connections_received": info.get("total_connections_received", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_ratio": round(
                    info.get("keyspace_hits", 0) / max(
                        info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                    ) * 100, 2
                ),
                "uptime_in_days": info.get("uptime_in_days", 0),
                "total_keys": sum(
                    int(db.get("keys", 0))
                    for db in info.get("db", {}).values()
                ),
            }
        except Exception as e:
            self._disable()
            return {"enabled": False, "error": str(e), "circuit_breaker_state": self._cb.state.value}
