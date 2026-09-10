"""
Celery Worker 启动脚本 (Windows 兼容)
用法: python start_celery.py
"""
import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

log_path = os.path.join(PROJECT_ROOT, "logs", "celery_worker.log")
pid_path = os.path.join(PROJECT_ROOT, "logs", ".celery_worker_heartbeat")

flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0

# 启动 Celery worker
cmd = [
    sys.executable, "-m", "celery",
    "-A", "src.celery_app",
    "worker",
    "--pool=solo",
    "--concurrency=1",
    "--without-mingle",
    "--without-gossip",
    "--loglevel=info",
]

print(f"Starting Celery worker: {' '.join(cmd)}")
print(f"Log: {log_path}")

with open(log_path, "w") as log:
    log.write(f"Starting Celery worker at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )

# 写入心跳
with open(pid_path, "w") as f:
    f.write(str(proc.pid))

print(f"Worker PID: {proc.pid}")
print("Waiting 10s for worker to initialize...")
time.sleep(10)

# 检查 Worker 是否正常运行
with open(log_path) as f:
    content = f.read()
    if "celery@" in content and "ready" in content.lower():
        print("Worker started successfully!")
    elif "error" in content.lower() or "traceback" in content.lower():
        print("Worker startup had issues:")
        for line in content.split("\n")[-10:]:
            print(f"  {line}")
    else:
        print("Worker started (check log for details)")

# 测试连接
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    # 看 Worker 是否注册了自己
    time.sleep(2)
    workers = [k.decode() for k in r.keys('celery-worker*')]
    if workers:
        print(f"Worker registered in Redis: {workers}")
    else:
        print("Worker not yet registered in Redis (may take longer)")
except Exception as e:
    print(f"Redis check: {e}")

print(f"\nTo check logs: tail -f {log_path}")
print(f"To stop: taskkill /F /PID {proc.pid}")
