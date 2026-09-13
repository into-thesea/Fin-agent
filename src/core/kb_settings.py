"""
知识库可编辑参数 —— 单一配置文件 data/kb_settings.json

为什么不用 .env: settings 是进程启动时导入的, 改完不重启不生效; 且 .env 里混着密钥。
为什么不用 SQLite: 这是单例配置不是记录集合, 存一行反而绕一层。

读取带进程内缓存 + mtime 失效 —— 手工改文件也能立刻生效, 不用重启。
"""
from __future__ import annotations

import os
import json
import time
import logging
import threading
from typing import Optional

from src.infra.paths import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULTS: dict = {
    "chunking": {"chunk_size": 350, "overlap": 70, "max_chunk_content": 300},
    "retrieval": {
        "top_k": 5,
        "mmr_enabled": True,
        "mmr_lambda": 0.5,
        "rrf_k": 60,
        "rerank_enabled": False,
        "rerank_model": "gte-rerank-v2",
        "rerank_pool": 20,
    },
}

_lock = threading.RLock()
_cache: Optional[dict] = None
_cache_mtime: float = 0.0


def path() -> str:
    """配置文件绝对路径。测试通过 monkeypatch 本函数指向临时文件。"""
    return os.path.join(DATA_DIR, "kb_settings.json")


def invalidate_cache() -> None:
    global _cache, _cache_mtime
    with _lock:
        _cache, _cache_mtime = None, 0.0


def _merge(section: str, raw: dict) -> dict:
    """以默认值为底, 用文件里的值覆盖 —— 文件缺字段不会让配置残缺"""
    out = dict(DEFAULTS.get(section, {}))
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in out:
                out[k] = v
    return out


def _with_meta(cfg: dict, updated_at: str = "", updated_by: str = "") -> dict:
    """补上元字段 —— updated_at/updated_by 是审计信息不是可编辑配置, 故不进 DEFAULTS"""
    cfg["updated_at"] = updated_at
    cfg["updated_by"] = updated_by
    return cfg


def _copy(cfg: dict) -> dict:
    """外传前复制: section 也要复制, 否则调用方改值会污染进程内缓存"""
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}


def _load() -> tuple[dict, str]:
    """返回 (配置, source)。文件不可读/损坏时回落默认值并记 ERROR。"""
    global _cache, _cache_mtime
    p = path()

    with _lock:
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            # 文件不存在: 正常的初始状态, 不是错误
            return _with_meta({s: dict(v) for s, v in DEFAULTS.items()}), "default"

        if _cache is not None and mtime == _cache_mtime:
            return _copy(_cache), "file"

        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):     # 合法 JSON 但不是对象, 同样算损坏
                raise ValueError("顶层不是 JSON 对象")
            cfg = {s: _merge(s, raw.get(s, {})) for s in DEFAULTS}
            cfg = _with_meta(
                cfg,
                str(raw.get("updated_at") or ""),
                str(raw.get("updated_by") or ""),
            )
        except Exception as e:
            # 损坏要大声 —— 否则"改了没生效"会被当成前端 bug 查半天
            logger.error("kb_settings.json 不可读, 回落代码默认值: %s", e)
            return _with_meta({s: dict(v) for s, v in DEFAULTS.items()}), "default"

        _cache, _cache_mtime = cfg, mtime
        return _copy(cfg), "file"


def get_all() -> tuple[dict, str]:
    return _load()


def get_chunking() -> dict:
    return _load()[0]["chunking"]


def get_retrieval() -> dict:
    return _load()[0]["retrieval"]


def save(section: str, values: dict, user: str = "") -> dict:
    """完整替换某个 section 的值并落盘。校验由 kb_validate 负责, 本函数只管存取。"""
    if section not in DEFAULTS:
        raise ValueError(f"未知配置段: {section}")
    cfg, _ = _load()
    cfg = {s: dict(v) for s, v in cfg.items()}
    cfg[section] = {**DEFAULTS[section], **values}
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["updated_by"] = user

    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)          # 原子替换, 避免半截文件
    invalidate_cache()
    return cfg
