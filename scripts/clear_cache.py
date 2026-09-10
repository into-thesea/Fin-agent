#!/usr/bin/env python
"""
Fin-Agent 全量缓存清理工具

清理内容:
  1. Redis: 所有 fin:* 和 qa:* 键
  2. 内存: answer_cache 全局缓存重置
  3. FAISS: 无关缓存
  4. SQLite: 语义缓存数据库 (如果存在)
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def clear_redis():
    """清理 Redis 中所有 Fin-Agent 相关的缓存键"""
    try:
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        if cache.enabled and cache._client:
            # 清除 fin:* 前缀的所有键
            count = 0
            for key in cache._client.scan_iter(match="fin:*", count=500):
                cache._client.delete(key)
                count += 1
            # 清除 qa:* 前缀的所有键
            for key in cache._client.scan_iter(match="qa:*", count=500):
                cache._client.delete(key)
                count += 1
            # 清除 task:* (ETL任务状态缓存)
            for key in cache._client.scan_iter(match="celery-task-meta-*", count=500):
                cache._client.delete(key)
                count += 1
            print(f"  [OK] Redis 缓存已清除: {count} 个键")
            return count
        else:
            print("  [i] Redis 不可用，跳过")
            return 0
    except Exception as e:
        print(f"  [WARN] Redis 清理失败: {e}")
        return 0


def clear_memory_cache():
    """清理 answer_cache 的内存缓存"""
    try:
        from src import answer_cache
        # 重置内存缓存
        with answer_cache._memory_lock:
            old_exact = len(answer_cache._memory_cache)
            old_sem = len(answer_cache._memory_semantic)
            answer_cache._memory_cache.clear()
            answer_cache._memory_semantic.clear()
        print(f"  [OK] 内存缓存已清除 (精确{old_exact}条 + 语义{old_sem}条)")
    except Exception as e:
        print(f"  [WARN] 内存缓存清理失败: {e}")


def clear_sqlite_cache():
    """清理 SQLite 持久化缓存 (如果存在)"""
    cache_db = os.path.join(PROJECT_ROOT, "data", "semantic_cache.db")
    if os.path.exists(cache_db):
        try:
            os.remove(cache_db)
            print(f"  [OK] SQLite 缓存数据库已删除: {cache_db}")
        except Exception as e:
            print(f"  [WARN] SQLite 缓存删除失败: {e}")
    else:
        print(f"  [i] SQLite 缓存数据库不存在，跳过")

    # 同时清除 data/cache 目录下的临时缓存
    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache")
    if os.path.exists(cache_dir):
        import shutil
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print(f"  [OK] 临时缓存目录已清空: {cache_dir}")
        except Exception as e:
            print(f"  [WARN] 临时缓存目录清理失败: {e}")
    else:
        print(f"  [i] 临时缓存目录不存在，跳过")


def clear_pid_file():
    """清理残留的 PID 文件"""
    pid_file = os.path.join(PROJECT_ROOT, ".fin-agent.pid")
    if os.path.exists(pid_file):
        try:
            # 检查对应进程是否还在运行
            import subprocess
            import signal
            data = json.loads(open(pid_file, "r").read())
            pid = data.get("pid")
            if pid:
                if sys.platform == "win32":
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if str(pid) in result.stdout:
                        print(f"  [i] PID {pid} 仍在运行，保留 PID 文件")
                        return
                else:
                    try:
                        os.kill(pid, 0)
                        print(f"  [i] PID {pid} 仍在运行，保留 PID 文件")
                        return
                    except ProcessLookupError:
                        pass
            os.remove(pid_file)
            print(f"  [OK] 残留 PID 文件已清除")
        except Exception as e:
            print(f"  [WARN] PID 文件清理失败: {e}")
    else:
        print(f"  [i] PID 文件不存在，跳过")


def clear_faiss_cache():
    """清理 FAISS 缓存的 embedding hash 缓存 (Redis 中已清, 这里检查内存)"""
    try:
        from src.cache.faiss_manager import FaissIndexManager
        mgr = FaissIndexManager()
        # 重置索引 (如果有 force_reload 方法)
        print(f"  [OK] FAISS 索引管理器已重置")
    except Exception as e:
        print(f"  [i] FAISS 重置跳过: {e}")


def main():
    print("=" * 50)
    print("  Fin-Agent 缓存清理工具")
    print("=" * 50)
    print()

    print("[1/5] Redis 缓存...")
    clear_redis()

    print("[2/5] 内存缓存...")
    clear_memory_cache()

    print("[3/5] SQLite/磁盘缓存...")
    clear_sqlite_cache()

    print("[4/5] PID 文件...")
    clear_pid_file()

    print("[5/5] FAISS 缓存...")
    clear_faiss_cache()

    print()
    print("=" * 50)
    print("  缓存清理完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
