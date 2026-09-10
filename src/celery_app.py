"""
Fin-Agent Celery 异步任务应用

Celery Worker 启动命令:
  celery -A src.celery_app worker --loglevel=info --concurrency=4

Celery Beat 定时任务 (可选):
  celery -A src.celery_app beat --loglevel=info

docker-compose 中已定义 worker 服务，自动启动。
"""

import os
import logging

from celery import Celery
from dotenv import load_dotenv

# 确保 .env 在 Worker 启动时加载
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from src.config import settings

# 读取配置
REDIS_HOST = settings.redis_host
REDIS_PORT = settings.redis_port
REDIS_PASSWORD = settings.redis_password

redis_url = f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/0"

# 创建 Celery 应用
celery_app = Celery(
    "fin_agent",
    broker=redis_url,
    backend=redis_url,
    include=["src.tasks.etl_tasks"],  # 自动发现任务模块
)

# Celery 配置
celery_app.conf.update(
    # 任务序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    # 任务结果过期时间 (1 天)
    result_expires=86400,
    # 任务软/硬超时 (10 分钟 / 15 分钟)
    task_soft_time_limit=600,
    task_time_limit=900,
    # 每个 Worker 最多处理 1000 个任务后重启 (防止内存泄漏)
    worker_max_tasks_per_child=1000,
    # 任务发送完成确认 (可靠)
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # 重试策略默认值
    task_default_retry_delay=60,
    task_max_retries=3,
    # 预取数量 (每个 Worker 一次预取 1 个任务，保证公平调度)
    worker_prefetch_multiplier=1,
    # 注解全局速率限制 (可选)
    # task_annotations = {'*': {'rate_limit': '10/m'}}
)

logger = logging.getLogger(__name__)
logger.info(
    "Celery 应用初始化完成",
    extra={"broker": redis_url.replace(REDIS_PASSWORD or "", "***") if REDIS_PASSWORD else redis_url},
)


# 用于兼容 celery worker 命令行
app = celery_app


# ──────────────────────────────────────────────
# Worker 健康检测
# ──────────────────────────────────────────────

# Celery Worker 心跳标记文件 (Windows 上 control.ping 不可靠)
_CELERY_WORKER_HEARTBEAT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", ".celery_worker_heartbeat"
)


def _touch_heartbeat(**kwargs):
    """创建/更新心跳文件 (在 worker 就绪时自动调用)"""
    try:
        os.makedirs(os.path.dirname(_CELERY_WORKER_HEARTBEAT), exist_ok=True)
        with open(_CELERY_WORKER_HEARTBEAT, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        logger.debug("心跳文件写入失败 (可忽略): %s", e)


# Worker 就绪时自动创建心跳文件
from celery.signals import worker_ready
worker_ready.connect(_touch_heartbeat)


def is_celery_worker_running(timeout: float = 2.0) -> bool:
    """
    检测是否有 Celery Worker 正在消费队列

    Windows 上 `control.ping()` 因 solo pool 限制会挂起，
    改用进程检测 + 心跳文件双重保障。

    Returns:
        True — 至少一个 Worker 在线
        False — 无 Worker
    """
    # 方式一: 心跳文件检测 (Worker 启动时创建)
    if os.path.exists(_CELERY_WORKER_HEARTBEAT):
        try:
            mtime = os.path.getmtime(_CELERY_WORKER_HEARTBEAT)
            import time
            if time.time() - mtime < 120:
                return True
        except OSError as e:
            logger.debug("心跳文件时间戳读取失败 (可忽略): %s", e)

    # 方式二: 进程命令行检测 (Windows 兜底)
    # 注意: worker 以 pythonw.exe 运行, tasklist 按 IMAGENAME 过滤匹配不到;
    # 需用 PowerShell 按命令行匹配 (含 celery + celery_app + worker 的唯一进程)
    try:
        import subprocess
        ps_cmd = (
            "Get-CimInstance Win32_Process -Filter \"name like 'python%'\" "
            "| Where-Object { $_.CommandLine -match 'celery' -and "
            "$_.CommandLine -match 'celery_app' -and $_.CommandLine -match 'worker' } "
            "| Measure-Object | Select-Object -ExpandProperty Count"
        )
        kwargs = {}
        if hasattr(subprocess, 'CREATE_NO_WINDOW'):
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            capture_output=True, encoding='utf-8', errors='replace',
            timeout=5, **kwargs,
        )
        count = result.stdout.strip()
        if result.returncode == 0 and count.isdigit() and int(count) > 0:
            return True
    except Exception as e:
        logger.debug("进程命令行检测异常 (可忽略): %s", e)

    # 方式三: control.ping (Linux/macOS 上有效)
    try:
        import platform
        if platform.system() != "Windows":
            result = celery_app.control.ping(timeout=timeout)
            return len(result) > 0
    except Exception as e:
        logger.debug("control.ping 失败 (可忽略): %s", e)

    return False
