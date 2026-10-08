"""
Fin-Agent 异常层次结构

所有业务异常继承 FinAgentError，API 层通过全局处理器捕获并返回统一格式。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FinAgentError(Exception):
    """Fin-Agent 应用异常基类"""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR",
                 status_code: int = 500, details: Optional[dict] = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigError(FinAgentError):
    """配置错误 (如缺少 API Key)"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="CONFIG_ERROR", status_code=500, details=details)


class LLMError(FinAgentError):
    """LLM 调用错误"""
    def __init__(self, message: str, code: str = "LLM_ERROR",
                 status_code: int = 502, details: Optional[dict] = None):
        super().__init__(message, code=code, status_code=status_code, details=details)


class RetrievalError(FinAgentError):
    """检索层错误"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="RETRIEVAL_ERROR", status_code=502, details=details)


class StorageError(FinAgentError):
    """存储层错误 (SQLite / Milvus / Pickle)"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="STORAGE_ERROR", status_code=500, details=details)


class CacheError(FinAgentError):
    """缓存层错误 (Redis / 语义缓存)"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="CACHE_ERROR", status_code=500, details=details)


class PipelineError(FinAgentError):
    """分析管道错误"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="PIPELINE_ERROR", status_code=500, details=details)


def error_response(error: FinAgentError, trace_id: str = "") -> dict:
    """生成统一的错误响应格式"""
    return {
        "success": False,
        "error": {
            "code": error.code,
            "message": error.message,
            "trace_id": trace_id,
        },
    }
