"""
pytest 全局配置与 Fixtures
"""

import os
import sys

import pytest

# 确保 src 可导入
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture(autouse=True)
def setup_test_env():
    """每个测试前设置测试环境变量"""
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    os.environ.setdefault("LOG_FORMAT", "text")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    os.environ.setdefault("LOG_FORMAT", "text")
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
    yield
