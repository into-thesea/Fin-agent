"""
轻量断路器 — 防止对故障服务的重复调用

状态机: CLOSED → OPEN (失败阈值) → HALF_OPEN (超时后) → CLOSED/OPEN

用法:
    cb = CircuitBreaker("redis", failure_threshold=3, recovery_timeout=30)
    try:
        result = cb.call(redis_client.ping)
    except CircuitBreakerOpenError:
        logger.warning("Redis 熔断中，降级")
"""

import time
import logging
from enum import Enum
from typing import Callable, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """断路器熔断异常 — 服务不可用"""
    pass


class CircuitBreaker:
    """
    轻量断路器

    当连续失败达到 failure_threshold 时熔断 (OPEN)，
    经过 recovery_timeout 秒后转为 HALF_OPEN 允许探测请求，
    探测成功则恢复 (CLOSED)，失败则保持 OPEN。
    """

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0, half_open_max_tries: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_tries = half_open_max_tries
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_tries = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_tries = 0
                logger.info("断路器 %s: OPEN → HALF_OPEN (超时恢复)", self.name)
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """在断路器保护下调用函数"""
        state = self.state
        if state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"断路器 {self.name} 已熔断")
        if state == CircuitState.HALF_OPEN:
            if self._half_open_tries >= self.half_open_max_tries:
                raise CircuitBreakerOpenError(
                    f"断路器 {self.name} HALF_OPEN 探测耗尽 ({self.half_open_max_tries}次)"
                )
            self._half_open_tries += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self._state == CircuitState.HALF_OPEN:
            logger.info("断路器 %s: HALF_OPEN → CLOSED (探测成功)", self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_tries = 0

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_tries >= self.half_open_max_tries:
                self._state = CircuitState.OPEN
                logger.warning("断路器 %s: HALF_OPEN → OPEN (探测耗尽)", self.name)
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning("断路器 %s: CLOSED → OPEN (%d次失败)",
                          self.name, self._failure_count)

    def reset(self):
        """手动重置断路器"""
        old = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_tries = 0
        if old != CircuitState.CLOSED:
            logger.info("断路器 %s: %s → CLOSED (手动重置)", self.name, old.value)

    @property
    def available(self) -> bool:
        """快速检查是否可用（不改变状态）"""
        return self.state != CircuitState.OPEN

    @property
    def failure_count(self) -> int:
        """当前连续失败次数"""
        return self._failure_count

    @property
    def half_open_tries(self) -> int:
        """当前 HALF_OPEN 下的探测次数"""
        return self._half_open_tries
