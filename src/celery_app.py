"""
Fin-Agent Celery 异步任务应用

Celery Worker 启动命令:
  celery -A src.celery_app worker --loglevel=info --concurrency=4

Celery Beat 定时任务 (可选):
  celery -A src.celery_app beat --loglevel=info

docker-compose 中已定义 worker 服务，自动启动。
"""

import logging
import os
import threading
import time

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
    include=["src.tasks.etl_tasks", "src.tasks.kb_tasks"],  # 自动发现任务模块
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

# Celery Worker 心跳标记文件 (Windows 上 control.ping 因 solo pool 不可靠)
_CELERY_WORKER_HEARTBEAT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", ".celery_worker_heartbeat"
)

_HEARTBEAT_INTERVAL = 30   # worker 每 30s 刷新一次
# 超过这个时长未刷新判定离线。取 3 倍刷新间隔: 足够容忍抖动, 又不至于让死掉的
# worker 装活太久(实测 180s 时, 进程已死但文件还"够新", 判定照样返回 True)。
_HEARTBEAT_MAX_AGE = 90


def _touch_heartbeat():
    """创建/更新心跳文件"""
    try:
        os.makedirs(os.path.dirname(_CELERY_WORKER_HEARTBEAT), exist_ok=True)
        with open(_CELERY_WORKER_HEARTBEAT, "w") as f:
            f.write(str(time.time()))
    except OSError as e:
        # 这里原来只记 debug 且函数体内用了未 import 的 time.time() → NameError 被
        # 静默吞掉, 心跳文件从未更新, worker 在线检测一直靠 spawn PowerShell 扫进程。
        logger.error("心跳文件写入失败 (Worker 在线检测会失准): %s", e)


_heartbeat_started = False


def _start_heartbeat(**kwargs):
    """启动后台刷新线程 (由 worker 启动信号触发, 幂等)。

    原来只在 worker_ready 里写一次 —— 而判定逻辑要求 120s 内刷新过,
    于是跑够两分钟的 worker 会被判成离线。
    """
    global _heartbeat_started
    if _heartbeat_started:
        return
    _heartbeat_started = True
    logger.info("Celery Worker 心跳已启动: %s", _CELERY_WORKER_HEARTBEAT)
    _touch_heartbeat()

    def _loop():
        while True:
            time.sleep(_HEARTBEAT_INTERVAL)
            _touch_heartbeat()

    threading.Thread(target=_loop, daemon=True, name="celery-heartbeat").start()


def _stop_heartbeat(**kwargs):
    """worker 正常退出时立刻删掉心跳文件。

    不删的话它会一直躺在磁盘上, 在 `_HEARTBEAT_MAX_AGE` 窗口内让已死的 worker
    继续被判为在线 —— 用户点上传, 任务投进队列却永远没人消费。
    (进程被强杀时收不到这个信号, 那种情况由 max_age 兜底。)
    """
    try:
        os.remove(_CELERY_WORKER_HEARTBEAT)
        logger.info("Celery Worker 心跳文件已清理")
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.error("心跳文件清理失败: %s", e)


from celery.signals import worker_init, worker_ready, worker_shutdown

# worker_init 在 worker 进程一开始就发, worker_ready 在就绪后发。
# 两个都接: 只依赖 worker_ready 时实测没触发(心跳文件长期不更新)。
worker_init.connect(_start_heartbeat)
worker_ready.connect(_start_heartbeat)
worker_shutdown.connect(_stop_heartbeat)


def is_celery_worker_running() -> bool:
    """
    检测是否有 Celery Worker 在线。

    只认心跳文件 —— 它由 worker 自己每 30s 刷新, 是「worker 进程活着」最直接的证据。
    原先还会 spawn 一次 PowerShell 扫进程命令行: 每次上传跑一次、批量上传 N 个文件
    就并发 N 次, 且按命令行正则匹配本身带误报风险。
    """
    try:
        mtime = os.path.getmtime(_CELERY_WORKER_HEARTBEAT)
    except OSError:
        logger.error("Celery Worker 心跳文件不存在: %s", _CELERY_WORKER_HEARTBEAT)
        return False

    age = time.time() - mtime
    if age < _HEARTBEAT_MAX_AGE:
        return True
    logger.error("Celery Worker 心跳已过期 %.0fs (阈值 %ds), 判定为离线",
                 age, _HEARTBEAT_MAX_AGE)
    return False
