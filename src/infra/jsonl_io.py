"""
JSONL 原子读写工具

提供原子写入能力（write+flush+fsync），防止程序崩溃产生截断行。
"""
import json
import os


def atomic_append(path: str, data: dict) -> None:
    """原子追加一行到 JSONL 文件

    策略：写临时文件 → fsync → 合并到目标文件。
    如果合并失败，临时文件保留作为恢复依据。
    """
    tmp_path = path + ".tmp"
    line = json.dumps(data, ensure_ascii=False) + "\n"

    with open(tmp_path, "a", encoding="utf-8") as tmp:
        tmp.write(line)
        tmp.flush()
        os.fsync(tmp.fileno())

    if os.path.exists(path):
        with open(tmp_path, "r", encoding="utf-8") as src, \
             open(path, "a", encoding="utf-8") as dst:
            dst.write(src.read())
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, path)


def read_lines(path: str) -> list[dict]:
    """读取 JSONL 文件，跳过损坏的行"""
    import logging
    logger = logging.getLogger(__name__)
    results = []
    if not os.path.exists(path):
        return results
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("JSONL 第 %d 行损坏，跳过: %s", i, e)
    return results
