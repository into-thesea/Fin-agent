"""知识库版本守卫 — 知识库更新后旧答案必须作废

没有这道守卫时, KB 更新后缓存的旧答案会在 TTL(1小时) 内继续被返回。
这是"内容改了, 但缓存没有依据知道该失效"的具体表现。
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import src.core.answer_cache as ac


def _read_manifest():
    with open(ac._MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(data):
    with open(ac._MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_manifest_path_is_project_data_dir():
    """清单在项目根 data/ 下, 不是 src/data/ (本文件 PROJECT_ROOT 少一层的坑)"""
    assert os.path.exists(ac._MANIFEST_PATH), "知识库清单不存在, 先跑 scripts/sync_kb.py"
    assert ac._MANIFEST_PATH.replace("\\", "/").endswith("/data/.kb_manifest.json")


def test_version_matches_manifest():
    assert ac._current_kb_version() == _read_manifest()["version"]


def test_cache_invalidated_when_kb_version_changes():
    """核心断言: 版本一变, 之前写进去的答案就查不到了"""
    original = _read_manifest()
    try:
        ac.invalidate(None)
        ac._kb_guard["seen"] = ac._current_kb_version()   # 基线对齐当前版本

        query = "知识库版本守卫单元测试问题"
        ac.set_cache(query, {"answer": "基于旧知识库的答案"})
        assert ac.get_cached(query) is not None, "前置条件: 刚写入的答案应能命中"

        _write_manifest({**original, "version": "TEST_VERSION_CHANGED"})
        time.sleep(0.05)                                  # 确保 mtime 变化被察觉

        assert ac._current_kb_version() == "TEST_VERSION_CHANGED"
        assert ac.get_cached(query) is None, "知识库版本变了, 旧答案必须作废"
    finally:
        _write_manifest(original)
        ac.invalidate(None)
        ac._kb_guard["mtime"] = 0.0
        ac._kb_guard["file_version"] = None
        ac._kb_guard["seen"] = None


def test_first_call_only_records_baseline():
    """进程内首次调用不应清缓存 (否则每次冷启动都白清一遍)"""
    ac._kb_guard["seen"] = None
    ac._invalidate_on_kb_change()
    assert ac._kb_guard["seen"] == ac._current_kb_version()
