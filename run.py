#!/usr/bin/env python
"""
Fin-Agent 4.0 — 一键启动入口 (增强版)

特性:
  - 端口自动检测: 启动前检查端口占用, 自动杀掉旧进程
  - PID 文件管理: `--stop` 优雅关闭, 无需手动 kill
  - 启动健康检查: 等待 /health 返回 200 才算启动成功
  - 后台运行: `--daemon` 模式脱离终端
  - 自动重启: `--restart` = stop + start
  - 零弹窗: 自动探测 pythonw.exe, 启动时无命令行窗口

用法:
    python run.py                      # 生产模式 (端口 8001)
    python run.py --dev                # 开发模式 (后端 8001 + 前端 3000)
    python run.py --port 9000          # 自定义端口
    python run.py --stop               # 关闭已运行的实例
    python run.py --restart            # 重启
    python run.py --daemon             # 后台运行 (不占用终端)
    python run.py --status             # 查看运行状态
"""

import os
import sys
import time
import json
import signal
import socket
import argparse
import subprocess
from pathlib import Path

_PROCFLAGS = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
_IS_WINDOWS = sys.platform == "win32"

PROJECT_ROOT = Path(__file__).parent.resolve()
CELERY_PID_FILE = PROJECT_ROOT / ".fin-celery.pid"

# ── 启动引导: 强制使用项目 .venv 的 python ──────────
# 直接 `python run.py` (系统全局 python) 可能缺项目依赖 (faiss/redis/python-multipart 等)。
# 若当前 python 不是 .venv 且 .venv 存在, 自动用 .venv 重新执行本脚本, 保证环境完整。
_VENV_PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if (
    _VENV_PY.exists()
    and os.path.abspath(sys.executable).lower() != str(_VENV_PY).lower()
):
    os.execv(str(_VENV_PY), [str(_VENV_PY), str(PROJECT_ROOT / "run.py")] + sys.argv[1:])


# ──────────────────────────────────────────────
# Celery Worker 管理
# ──────────────────────────────────────────────

def write_celery_pid(pid: int):
    """写入 Celery PID 文件"""
    data = {"pid": pid, "started_at": time.time()}
    CELERY_PID_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def read_celery_pid() -> dict | None:
    if not CELERY_PID_FILE.exists():
        return None
    try:
        return json.loads(CELERY_PID_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def remove_celery_pid():
    if CELERY_PID_FILE.exists():
        CELERY_PID_FILE.unlink()


def is_celery_running() -> bool:
    """检查 Celery Worker 是否运行中"""
    data = read_celery_pid()
    if not data:
        return False
    pid = data.get("pid")
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=_PROCFLAGS,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def stop_celery_worker(force: bool = False) -> bool:
    """停止 Celery Worker"""
    data = read_celery_pid()
    if not data:
        print("  [i] Celery Worker 未运行")
        return True
    pid = data["pid"]
    print(f"  [STOP] 关闭 Celery Worker (PID {pid})...")
    if kill_pid_tree(pid, force=True):
        remove_celery_pid()
        print(f"  [OK] Celery Worker 已关闭")
        return True
    return False


def run_celery_worker(concurrency: int = 2):
    """启动 Celery Worker"""
    stop_celery_worker(force=True)

    print(f"  [START] 启动 Celery Worker (并发={concurrency})...")
    log_path = PROJECT_ROOT / "logs" / "celery_worker.log"
    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    cmd = [
        sys.executable, "-X", "utf8", "-m", "celery",
        "-A", "src.celery_app", "worker",
        "--loglevel=info", "--concurrency", str(concurrency),
        "--pool", "solo",  # Windows 兼容: 避免 billiard 多进程崩溃
    ]
    process = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    write_celery_pid(process.pid)
    print(f"  [OK] Celery Worker 已启动 (PID {process.pid})")
    print(f"      日志: {log_path}")

    time.sleep(3)
    if process.poll() is not None:
        print(f"  [!WARN] Celery Worker 已异常退出 (code={process.returncode})")
        remove_celery_pid()
        try:
            for line in log_path.read_text("utf-8").strip().splitlines()[-5:]:
                print(f"     {line}")
        except Exception:
            pass
        return False
    return True


PID_FILE = PROJECT_ROOT / ".fin-agent.pid"


# ──────────────────────────────────────────────
# 零弹窗启动 — 自动探测 pythonw.exe
# ──────────────────────────────────────────────

def _find_pythonw() -> str | None:
    """查找 pythonw.exe (无窗口 Python)"""
    if not _IS_WINDOWS:
        return None
    # 和 python.exe 同目录
    py_dir = Path(sys.executable).parent
    candidates = [
        py_dir / "pythonw.exe",
        Path(sys.prefix) / "pythonw.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _relaunch_with_pythonw(force: bool = False):
    """
    自动检测 pythonw.exe, 将自身重新启动为无窗口模式.

    当用户从 Explorer 双击 run.py (或 python run.py --daemon) 时,
    python.exe 是控制台程序, 一定会弹命令行窗口.
    此函数检测到 pythonw.exe 存在, 就把自己重新启动为 pythonw.exe,
    实现「双击无弹窗」.
    """
    if not _IS_WINDOWS:
        return
    if "pythonw" in sys.executable.lower():
        return  # 已经是 pythonw, 无需重启动
    if "--no-relaunch" in sys.argv:
        return  # 防止无限循环
    if not force:
        return  # 只有调用方要求时才重启动

    pythonw = _find_pythonw()
    if not pythonw:
        return

    # 过滤掉 --no-relaunch (如果之前从别处传来)
    args = [a for a in sys.argv[1:] if a != "--no-relaunch"]

    try:
        subprocess.Popen(
            [pythonw, __file__, *args, "--no-relaunch"],
            cwd=str(PROJECT_ROOT),
            creationflags=_PROCFLAGS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return  # 重启动失败 → 继续用当前进程
    # 父进程 (python.exe) 立即退出, 命令行窗口随之关闭
    sys.exit(0)


# ──────────────────────────────────────────────
# 端口工具
# ──────────────────────────────────────────────

def is_port_in_use(port: int) -> bool:
    """检测端口是否被占用 (双重确认: bind + netstat)"""
    # 方法 1: socket.bind 探测
    bind_fail = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
    except OSError:
        bind_fail = True

    # 方法 2: netstat 查 LISTENING 进程 (弥补 SO_REUSEADDR 绕过 bind 检测)
    pid = get_pid_by_port(port)

    # 方法 3: 尝试连接 (服务已启动时)
    can_connect = False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            can_connect = True
    except (ConnectionRefusedError, socket.timeout, OSError):
        pass

    return bind_fail or pid is not None or can_connect


def get_pid_by_port(port: int) -> int | None:
    """查找占用端口的进程 PID (跨平台)"""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True,
                encoding="gbk", errors="replace", timeout=5,
                creationflags=_PROCFLAGS,
            )
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and f":{port}" in parts[1]:
                    state = parts[3] if len(parts) > 3 else ""
                    if state == "LISTENING" or "LISTEN" in state:
                        return int(parts[-1])
            # 再查一次带 LISTENING 的格式
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts[-1].isdigit():
                        return int(parts[-1])
        except Exception:
            return None
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                return int(result.stdout.strip().split("\n")[0])
        except Exception:
            try:
                result = subprocess.run(
                    ["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    return int(result.stdout.strip())
            except Exception:
                pass
    return None


def kill_pid(pid: int, force: bool = False) -> bool:
    """优雅终止进程, 失败则强制 (只杀指定 PID)"""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F" if force else "/PID", str(pid)],
                           capture_output=True, timeout=5, creationflags=_PROCFLAGS)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return True  # 进程已死
            if not force:
                time.sleep(1)
                try:
                    os.kill(pid, 0)  # 进程还在?
                    subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=5)
                except ProcessLookupError:
                    pass
        return True
    except Exception:
        return False


def kill_pid_tree(pid: int, force: bool = False) -> bool:
    """
    终止进程及其所有子进程 (进程树)

    Windows: taskkill /T (递归杀子树)
    Linux/mac: 递归查找子进程并逐个 kill
    """
    if sys.platform == "win32":
        flag = "/F" if force else ""
        try:
            # /T = 杀进程树, /PID = 指定 PID
            subprocess.run(
                ["taskkill", "/T", flag, "/PID", str(pid)],
                capture_output=True, timeout=5, creationflags=_PROCFLAGS,
            )
            return True
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    else:
        # Linux: 递归杀子树
        import subprocess as sp
        try:
            # 获取子进程列表
            result = sp.run(
                ["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=5
            )
            child_pids = [int(p) for p in result.stdout.strip().split()
                         if p.strip().isdigit()]
            # 先杀子进程
            for cpid in child_pids:
                kill_pid_tree(cpid, force=force)
            # 再杀自己
            return kill_pid(pid, force=force)
        except Exception:
            return kill_pid(pid, force=force)


def ensure_port_free(port: int, force: bool = False) -> bool:
    """
    确保端口可用.
    如果端口被占用, 尝试杀掉占用进程.
    Returns: True=端口可用, False=无法释放
    """
    if not is_port_in_use(port):
        return True

    pid = get_pid_by_port(port)
    if pid:
        proc_name = ""
        try:
            if sys.platform == "win32":
                r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                   capture_output=True, text=True, timeout=5, creationflags=_PROCFLAGS)
                for line in r.stdout.splitlines():
                    if str(pid) in line:
                        proc_name = line.split()[0]
                        break
            else:
                r = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                                   capture_output=True, text=True, timeout=5)
                proc_name = r.stdout.strip()
        except Exception:
            proc_name = f"PID={pid}"

        if force:
            print(f"  [KILL] 杀掉进程树 {proc_name} (PID {pid})...")
            kill_pid_tree(pid, force=True)
            time.sleep(1)
            if is_port_in_use(port):
                # 可能 TIME_WAIT, 直接尝试启动
                print(f"  [OK] 尝试直接绑定 (旧进程已死)")
                return True
            print(f"  [OK] 端口 {port} 已释放")
            return True
        else:
            print(f"  [!WARN] 端口 {port} 被 {proc_name} (PID {pid}) 占用")
            print(f"  使用 --force 自动杀掉, 或先运行 python run.py --stop")
            return False
    else:
        # 端口占用但找不到进程 -> TIME_WAIT (自己的旧进程刚死)
        # 等最多 5 秒, 然后直接尝试启动 (TIME_WAIT 通常不影响新进程绑定)
        print(f"  [WAIT] 端口 {port} TIME_WAIT (旧进程残留), 等待释放...", end="", flush=True)
        for i in range(5):
            time.sleep(1)
            if not is_port_in_use(port):
                print(f"\r  [OK] 端口 {port} 已释放{' '*40}")
                return True
            print(".", end="", flush=True)
        # 即使 TIME_WAIT 还在, 也尝试启动 (Windows 通常允许 reuse)
        print(f"\r  [OK] 尝试直接绑定 (TIME_WAIT, 无进程占用){' '*20}")
        return True


# ──────────────────────────────────────────────
# PID 文件管理
# ──────────────────────────────────────────────

def write_pid(port: int):
    """写入 PID 文件"""
    data = {
        "pid": os.getpid(),
        "port": port,
        "started_at": time.time(),
    }
    PID_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def read_pid() -> dict | None:
    """读取 PID 文件"""
    if not PID_FILE.exists():
        return None
    try:
        return json.loads(PID_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def remove_pid():
    """删除 PID 文件"""
    if PID_FILE.exists():
        PID_FILE.unlink()


def _clean_stale_pid():
    """清理过期 PID 文件: PID 文件存在但进程已死时自动删除"""
    data = read_pid()
    if not data:
        return
    pid = data.get("pid")
    port = data.get("port", 0)
    if pid and not is_port_in_use(port):
        # 端口空闲 → 进程已死 → 清理脏 PID 文件
        remove_pid()


def stop_running_instance() -> bool:
    """通过 PID 文件关闭运行中的实例 (连带子进程全部清除)"""
    data = read_pid()
    if not data:
        print("  [i] 没有找到运行中的 Fin-Agent 实例 (PID 文件不存在)")
        # 回退: 检查默认端口
        for port in [8001, 8000]:
            if is_port_in_use(port):
                pid = get_pid_by_port(port)
                if pid:
                    print(f"  [STOP] 发现端口 {port} 上的进程 (PID {pid}), 正在关闭...")
                    # 杀进程树 (含 uvicorn worker 子进程)
                    if kill_pid_tree(pid):
                        print(f"  [OK] 已关闭 (含子进程)")
                        remove_pid()
                        return True
                    if kill_pid(pid, force=True):
                        print(f"  [OK] 已强制关闭")
                        remove_pid()
                        return True
        return False

    pid = data["pid"]
    port = data.get("port", 8001)
    print(f"  [STOP] 关闭实例 (PID {pid}, 端口 {port})...")

    killed = False
    # 先优雅杀进程树 (含 uvicorn worker)
    if kill_pid_tree(pid):
        killed = True

    # 如果进程树杀完但端口还被占用 → 端口上有残留进程
    time.sleep(1)
    if is_port_in_use(port):
        leftover = get_pid_by_port(port)
        if leftover and leftover != pid:
            print(f"  [KILL] 端口 {port} 还有残留进程 (PID {leftover}), 强制清理...")
            kill_pid_tree(leftover, force=True)
            killed = True

    if killed:
        remove_pid()
        print(f"  [OK] 已关闭")
    else:
        # 优雅失败 → 强制杀树
        print(f"  [FAIL] 无法关闭 PID {pid}, 尝试强制杀进程树...")
        if kill_pid_tree(pid, force=True):
            remove_pid()
            print(f"  [OK] 已强制关闭进程树")
            return True

        # 树也杀不掉 → 逐个端口清理
        for pport in [port, 8001, 8000]:
            pid2 = get_pid_by_port(pport)
            if pid2:
                kill_pid(pid2, force=True)
        remove_pid()
        return True
    return True


# ──────────────────────────────────────────────
# 前端构建
# ──────────────────────────────────────────────

def check_frontend_build() -> bool:
    """检查前端是否已构建"""
    return (PROJECT_ROOT / "frontend" / "dist" / "index.html").is_file()


def build_frontend() -> bool:
    """构建前端"""
    print("  [BUILD] 正在构建前端...")
    result = subprocess.run(
        ["npx", "vite", "build"],
        cwd=str(PROJECT_ROOT / "frontend"),
        capture_output=True, text=True, timeout=120,
        creationflags=_PROCFLAGS,
    )
    if result.returncode == 0:
        print("  [OK] 前端构建完成")
        return True
    else:
        print("  [FAIL] 前端构建失败:")
        for line in result.stderr.splitlines()[-10:]:
            print(f"     {line}")
        return False


# ──────────────────────────────────────────────
# 健康检查
# ──────────────────────────────────────────────

def wait_for_health(port: int, timeout: int = 30) -> bool:
    """等待服务健康检查通过"""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    print(f"  [WAIT] 等待服务就绪 (/{timeout}s)...", end="", flush=True)

    for _ in range(timeout):
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                print(f"\r  [OK] 服务就绪: {url}")
                return True
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout):
            pass
        print(".", end="", flush=True)
        time.sleep(1)

    print(f"\r  [!WARN] 健康检查超时 (可能仍需数秒完成预热)")
    return False


# ──────────────────────────────────────────────
# 服务管理
# ──────────────────────────────────────────────

def run_production(port: int, force: bool = False, daemon: bool = False):
    """生产模式: FastAPI 托管前后端 (单端口)"""
    # 1. 端口检测 — daemon 模式自动 --force 清理旧进程
    if daemon and not force:
        force = True  # 后台模式自动清理, 无需用户额外加 --force
    print(f"  [SCAN] 检测端口 {port}...")
    if not ensure_port_free(port, force=force):
        sys.exit(1)

    # 2. 前端检查
    if not check_frontend_build():
        print("  [!WARN] 未检测到前端构建产物 (frontend/dist)")
        try:
            reply = input("  是否立即构建前端? (y/n): ").lower()
        except (EOFError, OSError):
            reply = "n"  # 无控制台 (pythonw) 时跳过
        if reply == "y":
            if not build_frontend():
                sys.exit(1)
        else:
            print("  [i] 仅启动 API 模式 (无前端界面)")

    # 3. 启动服务
    print(f"\n  [START] 启动 Fin-Agent 4.0")
    print(f"     访问 http://localhost:{port}")
    print(f"     API      http://localhost:{port}/docs")
    if daemon:
        print(f"     模式: 后台运行 (--daemon)")
    print()

    cmd = [
        sys.executable, "-X", "utf8", "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", "1",
        "--loop", "asyncio",
    ]

    if daemon:
        _run_daemon(cmd, port)
    else:
        _run_foreground(cmd, port)


def run_dev(port: int, force: bool = False):
    """开发模式: 后端 + 独立前端开发服务器"""
    # 端口检测
    print(f"  [SCAN] 检测端口 {port}...")
    if not ensure_port_free(port, force=force):
        sys.exit(1)

    processes = []
    cleanup_done = False

    def cleanup(sig=None, frame=None):
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        print("\n  [STOP] 正在关闭服务...")
        for p in processes:
            if p.poll() is None:
                if sys.platform == "win32":
                    # 杀进程树
                    subprocess.run(
                        ["taskkill", "/T", "/F", "/PID", str(p.pid)],
                        capture_output=True, timeout=5, creationflags=_PROCFLAGS,
                    )
                else:
                    p.terminate()
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        # 端口兜底清理
        time.sleep(0.5)
        for pport in [port, 3000]:
            if is_port_in_use(pport):
                leftover = get_pid_by_port(pport)
                if leftover:
                    kill_pid(leftover, force=True)
        remove_pid()
        print("  [OK] 已关闭")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 启动后端
    print(f"  [SETUP] 启动后端 (端口 {port}, 热重载)...")
    backend = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "uvicorn",
         "src.api.main:app", "--host", "0.0.0.0",
         "--port", str(port), "--reload"],
        cwd=str(PROJECT_ROOT),
        creationflags=_PROCFLAGS,
    )
    processes.append(backend)
    write_pid(port)
    time.sleep(2)

    # 启动前端
    frontend_dir = PROJECT_ROOT / "frontend"
    if (frontend_dir / "package.json").exists():
        print(f"  [SETUP] 启动前端 (端口 3000)...")
        frontend = subprocess.Popen(
            ["npx", "vite", "--host", "0.0.0.0", "--port", "3000"],
            cwd=str(frontend_dir),
            creationflags=_PROCFLAGS,
        )
        processes.append(frontend)
    else:
        print("  [!WARN] 未找到前端项目, 跳过")

    # 健康检查
    wait_for_health(port)

    print(f"\n  [OK] 开发模式已启动:")
    print(f"     后端 API:  http://localhost:{port}")
    print(f"     前端界面:  http://localhost:3000")
    print(f"     API 文档:  http://localhost:{port}/docs")
    print(f"\n     按 Ctrl+C 停止所有服务")

    try:
        while all(p.poll() is None for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


def _run_foreground(cmd: list, port: int):
    """前台运行, Ctrl+C 杀进程树 + 释放端口"""
    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), creationflags=_PROCFLAGS)
    write_pid(port)

    def _on_exit(sig=None, frame=None):
        pid = process.pid
        if process.poll() is None:
            # 杀进程树 (uvicorn 主进程 + worker 子进程)
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5, creationflags=_PROCFLAGS,
                )
            else:
                process.terminate()
                process.wait(timeout=5)
        # 确保端口释放
        time.sleep(0.5)
        if is_port_in_use(port):
            leftover = get_pid_by_port(port)
            if leftover:
                kill_pid(leftover, force=True)
        remove_pid()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    # 健康检查
    wait_for_health(port)

    try:
        process.wait()
    except KeyboardInterrupt:
        _on_exit()
    finally:
        # 进程退出后等待 TIME_WAIT 释放
        time.sleep(2)
        for _ in range(10):
            if not is_port_in_use(port):
                break
            time.sleep(1)
        leftover = get_pid_by_port(port)
        if leftover:
            kill_pid(leftover, force=True)
        remove_pid()


def _run_daemon(cmd: list, port: int):
    """后台运行"""
    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)
    log_path = PROJECT_ROOT / "logs" / "uvicorn_daemon.log"
    log_file = open(log_path, "a", encoding="utf-8")

    if sys.platform == "win32":
        process = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
    else:
        log_file.close()
        pid = os.fork()
        if pid > 0:
            print(f"  [OK] 服务已在后台启动 (PID {pid})")
            write_pid(port)
            return
        os.setsid()
        log_file = open(log_path, "a", encoding="utf-8")
        process = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdin=open(os.devnull, "r"),
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        process.wait()

    print(f"  [OK] 服务已在后台启动 (PID {process.pid})")
    write_pid(port)
    wait_for_health(port)


# ──────────────────────────────────────────────
# 状态查询
# ──────────────────────────────────────────────

def show_status():
    """显示当前运行状态"""
    data = read_pid()
    if data and is_port_in_use(data.get("port", 0)):
        uptime = time.time() - data["started_at"]
        print(f"  [OK] 服务运行中")
        print(f"     PID:   {data['pid']}")
        print(f"     端口:  {data['port']}")
        print(f"     运行:  {uptime:.0f} 秒")
        print(f"     访问:  http://localhost:{data['port']}")
    else:
        # PID 文件过期或不存在, 检查端口
        alive = False
        for port in [8001, 8000]:
            if is_port_in_use(port):
                pid = get_pid_by_port(port)
                if pid:
                    print(f"  [!] 发现端口 {port} 上的进程 (PID {pid}), 但 PID 文件丢失")
                    print(f"     运行 python run.py --stop 清理")
                    alive = True
        if not alive:
            print(f"  [i] Fin-Agent 未运行")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main():
    # ── 提前扫描: 是否已有 --no-relaunch (从 pythonw 子进程传来) ──
    _no_relaunch = "--no-relaunch" in sys.argv

    parser = argparse.ArgumentParser(
        description="Fin-Agent 4.0 启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python run.py                        # 生产模式 (端口 8001)
  python run.py --port 9000            # 自定义端口
  python run.py --dev                  # 开发模式 (热重载)
  python run.py --stop                 # 关闭运行中的实例
  python run.py --restart              # 重启
  python run.py --daemon               # 后台运行 (不占用终端, 无弹窗)
  python run.py --silent               # 安静模式 (无命令行窗口, 等同于 --daemon)
  python run.py --status               # 查看状态
  python run.py --force                # 强制覆盖端口
        """,
    )
    parser.add_argument("--dev", action="store_true", help="开发模式 (前后端分离, 热重载)")
    parser.add_argument("--port", type=int, default=8001, help="后端端口 (默认 8001)")
    parser.add_argument("--stop", action="store_true", help="关闭运行中的实例")
    parser.add_argument("--restart", action="store_true", help="重启 (stop + start)")
    parser.add_argument("--daemon", action="store_true", help="后台运行 (不占用终端, 无弹窗)")
    parser.add_argument("--silent", action="store_true", help="安静模式 (别名 --daemon, 无命令行窗口)")
    parser.add_argument("--force", action="store_true", help="端口冲突时自动杀掉占用进程")
    parser.add_argument("--status", action="store_true", help="查看运行状态")
    parser.add_argument("--celery", action="store_true",
                        help="启动 Celery Worker (现在是默认行为, 保留此参数仅为兼容)")
    parser.add_argument("--no-celery", action="store_true",
                        help="不启动 Celery Worker (此时上传会被拒绝 —— ETL 强依赖 worker)")
    parser.add_argument("--celery-concurrency", type=int, default=2, help="Celery 并发数 (默认 2)")
    parser.add_argument("--celery-stop", action="store_true", help="停止 Celery Worker")
    parser.add_argument("--build", action="store_true", help="仅构建前端")
    parser.add_argument("--no-relaunch", action="store_true",
                        help=argparse.SUPPRESS)  # 隐藏参数, 防止 pythonw 无限循环

    # ── 零弹窗重启动 ──
    # 如果用户用 python.exe 启动, 且带了 --daemon/--silent,
    # 则自动切换到 pythonw.exe (无窗口) 重新启动.
    # 先做简单解析以识别模式 (不需要完整 parser)
    need_silent = any(a in sys.argv for a in ("--daemon", "--silent"))
    if need_silent and not _no_relaunch:
        _relaunch_with_pythonw(force=True)

    args = parser.parse_args()

    # --silent 是 --daemon 的别名
    if args.silent:
        args.daemon = True

    # 启动前清理过期 PID 文件 (无对应进程时自动删除)
    _clean_stale_pid()

    # 状态查询
    if args.status:
        show_status()
        return

    # 关闭
    if args.stop:
        stop_celery_worker()
        stop_running_instance()
        return

    # 仅停止 Celery
    if args.celery_stop:
        stop_celery_worker()
        return

    # 仅构建前端
    if args.build:
        build_frontend()
        return

    # 重启
    if args.restart:
        print("  [RELOAD] 重启 Fin-Agent...")
        stop_celery_worker()
        stop_running_instance()
        time.sleep(2)

    # 启动
    print("=" * 50)
    print("  Fin-Agent 4.0")
    print("=" * 50)

    # 启动 Celery Worker —— **默认启动**。
    # 上传/ETL 现在强依赖 worker(没有 worker 时 knowledge 路由会直接拒绝上传,
    # 不再降级到同步线程)。原先只在显式 --celery 时才启动, 于是默认启动下
    # 用户点「批量上传」会直接吃到「ETL Worker 未运行」。
    # 注意 dev 模式也要起 —— 开发时同样需要上传。
    if not args.no_celery:
        run_celery_worker(concurrency=args.celery_concurrency)

    # 启动 API
    if args.dev:
        run_dev(args.port, force=args.force)
    else:
        run_production(args.port, force=args.force, daemon=args.daemon)

    # 如果程序走到这里 (foreground uvicorn 退出) → 清理 Celery
    if args.celery and not args.dev:
        stop_celery_worker()


if __name__ == "__main__":
    main()
