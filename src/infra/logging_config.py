"""
Fin-Agent 结构化日志系统

特性:
  - JSON 格式输出 (可通过 LOG_FORMAT=json 启用)
  - 全链路追踪: trace_id 贯穿每次查询
  - 上下文注入: query, user_id, latency 等自动附加
  - 敏感信息过滤: 自动遮盖 API Key、密码等
  - 多级别支持: DEBUG / INFO / WARNING / ERROR

用法:
  from src.infra.logging_config import get_logger, set_trace_id

  logger = get_logger(__name__)
  set_trace_id("req_abc123")
  logger.info("查询开始", extra={"query": "稳盈添利30天收益", "user": "u_001"})
"""

import os
import json
import logging
import logging.handlers
import time
import re
from contextvars import ContextVar
from typing import Optional

from src.config import settings

# ──────────────────────────────────────────────
# 配置 (从 Pydantic Settings 读取)
# ──────────────────────────────────────────────

LOG_LEVEL = settings.log_level.upper()
LOG_FORMAT = settings.log_format  # "json" | "text"
LOG_DIR = settings.log_dir
LOG_MAX_BYTES = settings.log_max_bytes  # 10MB
LOG_BACKUP_COUNT = settings.log_backup_count

SENSITIVE_PATTERNS = [
    (r'(api_key["\']?\s*[:=]\s*["\']?)[^"\'\s]+', r'\1***REDACTED***'),
    (r'(password["\']?\s*[:=]\s*["\']?)[^"\'\s]+', r'\1***REDACTED***'),
    (r'(token["\']?\s*[:=]\s*["\']?)[^"\'\s,}]+', r'\1***REDACTED***'),
]

# 上下文变量 (trace_id) — asyncio 原生支持，子线程需显式传递
_trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')


def set_trace_id(trace_id: Optional[str]):
    """设置当前上下文的 trace_id"""
    if trace_id:
        _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """获取当前上下文的 trace_id (未设置则自动生成)"""
    trace_id = _trace_id_var.get('')
    if not trace_id:
        trace_id = f"gen_{int(time.time() * 1000000) % 0xFFFFFFFF:08x}"
        _trace_id_var.set(trace_id)
    return trace_id


def reset_trace_id():
    """重置 trace_id"""
    _trace_id_var.set('')


def get_trace_id_snapshot() -> str:
    """获取当前 trace_id 的快照（用于传递到子线程）"""
    return _trace_id_var.get('')


# ──────────────────────────────────────────────
# JSON 格式化器
# ──────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """结构化 JSON 日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}.{record.msecs:03.0f}Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
        }

        # 注入 extra 字段
        # ttfb_ms / stream: 流式请求的首 token 时间与流式标记 —— 有它们才能把 SSE 的
        # 中间件耗时(到响应对象返回, ~1ms)与真实首字时间分开算(见 metrics_store 文件头)
        for key in ("query", "user_id", "latency_ms", "tokens", "cache_hit",
                     "entity", "task_id", "doc_id", "status", "duration", "stage",
                     "ttfb_ms", "stream"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        # 异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # 敏感信息过滤
        message = log_entry["message"]
        for pattern, replacement in SENSITIVE_PATTERNS:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        log_entry["message"] = message

        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """文本格式 (用于开发环境)"""

    def format(self, record: logging.LogRecord) -> str:
        trace = get_trace_id()
        record.trace_id = trace

        # 安全获取消息 (处理 args 与 format string 不匹配的情况)
        try:
            msg = record.getMessage()
        except (TypeError, ValueError):
            msg = record.msg
            if record.args:
                msg = f"{msg} {record.args}"
        for pattern, replacement in SENSITIVE_PATTERNS:
            msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
        record.msg = msg

        return super().format(record)


# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────

_configured = False


def configure_logging():
    """
    全局日志配置:
      - 控制台输出: JSON 或 彩色文本
      - 文件输出: JSON 轮转日志 (10MB, 保留 7 天)
    """
    global _configured
    if _configured:
        return
    _configured = True

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # 清除已有 handlers (避免重复配置)
    root_logger.handlers.clear()

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)

    if LOG_FORMAT == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(trace_id)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    root_logger.addHandler(console_handler)

    # 2. 文件 Handler (轮转)
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "fin-agent.log"),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # 3. 错误日志单独文件
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "fin-agent-error.log"),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    # 抑制第三方库的 DEBUG 日志
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger = get_logger(__name__)
    logger.info(
        "日志系统初始化",
        extra={
            "format": LOG_FORMAT,
            "level": LOG_LEVEL,
            "dir": LOG_DIR,
            "max_bytes": LOG_MAX_BYTES,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """获取带 trace_id 支持的 Logger"""
    return logging.getLogger(name)


# 应用启动时自动配置
configure_logging()
