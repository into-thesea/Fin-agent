"""
Fin-Agent 统一配置管理

使用 Pydantic Settings 替代散落的 os.getenv() 调用。
所有配置项集中管理，启动时自动校验。

用法:
    from src.config import settings
    redis_host = settings.redis_host
"""

import os

from pydantic_settings import BaseSettings
from typing import Optional


# ── 环境检测 ──
_app_env = os.getenv("APP_ENV", "development").lower()
_env_file_map = {
    "development": ".env",
    "dev": ".env",
    "production": ".env.prod",
    "prod": ".env.prod",
    "test": ".env.test",
}
_selected_env = _env_file_map.get(_app_env, ".env")

# 如果指定的环境文件不存在，回退到 .env
if not os.path.exists(_selected_env):
    _selected_env = ".env"


class Settings(BaseSettings):
    # ── 环境 ──
    app_env: str = "development"

    # ── LLM ──
    llm_provider: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_api_base: str = "https://api.deepseek.com"
    cheap_model: str = "gemini-2.5-flash"
    dashscope_api_key: str = ""

    # ── Gemini ──
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_cheap_model: Optional[str] = None
    gemini_fallback_models: str = ""

    # ── Redis ──
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: Optional[str] = None

    # ── 模型 ──
    model_cache_dir: str = "./models/"
    transformers_offline: int = 1
    hf_hub_offline: int = 1

    # ── 日志 ──
    log_level: str = "INFO"
    log_format: str = "json"
    log_dir: str = "./logs/"
    log_max_bytes: int = 10_485_760  # 10MB
    log_backup_count: int = 7

    # ── Worker ──
    max_workers: int = 2
    max_queue_size: int = 10
    default_timeout: int = 120
    shutdown_timeout: int = 30

    # ── 缓存 ──
    cache_ttl: int = 3600
    max_entries: int = 500
    l2_threshold: float = 0.50
    l3_threshold: float = 0.92

    # ── API ──
    sync_timeout: int = 120
    stream_timeout: int = 150

    # ── 智能客服 ──
    cs_brand_name: str = "智能客服"
    business_api_base_url: str = ""          # 业务 API 基地址 (真实 ERP 接入时使用)
    kb_extra_doc_types: str = ""             # 客服知识库额外文档类型 (逗号分隔: docx,xlsx)
    user_auth_mode: str = "header"           # header: 从 X-User-Id 读当前用户; none: 不做归属校验
    handoff_sla: int = 60                    # 人工转接响应 SLA 秒 (占位, Phase 3 使用)

    # ── 认证 (Phase 4) ──
    jwt_secret: str = "dev-secret-change-me"  # 生产环境务必通过环境变量覆盖
    token_ttl: int = 86400                   # token 有效期秒 (默认 1 天)
    admin_username: str = "admin"
    admin_password: str = ""                 # 为空时 login 拒绝 (演示模式仍可走前端 demo)
    user_username: str = "user"              # 普通用户账号 (双端: user=仅客服对话)
    user_password: str = ""

    model_config = {
        "env_file": _selected_env,
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # 兼容历史遗留环境变量 (如 NEO4J_*), 不报错
    }


settings = Settings()
