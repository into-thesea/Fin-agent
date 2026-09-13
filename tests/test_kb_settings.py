import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pytest
from src.core import kb_settings


@pytest.fixture(autouse=True)
def temp_settings(tmp_path, monkeypatch):
    """每个用例都指向临时配置文件, 不碰真实 data/kb_settings.json"""
    monkeypatch.setattr(kb_settings, "path", lambda: str(tmp_path / "kb_settings.json"))
    kb_settings.invalidate_cache()
    yield
    kb_settings.invalidate_cache()


def test_returns_defaults_when_file_missing():
    cfg, source = kb_settings.get_all()
    assert source == "default"
    assert cfg["chunking"]["chunk_size"] == kb_settings.DEFAULTS["chunking"]["chunk_size"]


def test_save_then_read_back():
    kb_settings.save("retrieval", {**kb_settings.DEFAULTS["retrieval"], "top_k": 9}, user="admin")
    cfg, source = kb_settings.get_all()
    assert source == "file"
    assert cfg["retrieval"]["top_k"] == 9


def test_cache_invalidated_after_external_edit():
    """模拟别人手工改了文件 —— 不需要重启进程就能读到新值"""
    kb_settings.save("retrieval", {**kb_settings.DEFAULTS["retrieval"], "top_k": 7}, user="admin")
    assert kb_settings.get_retrieval()["top_k"] == 7
    p = kb_settings.path()
    raw = json.load(open(p, encoding="utf-8"))
    raw["retrieval"]["top_k"] = 11
    json.dump(raw, open(p, "w", encoding="utf-8"))
    assert kb_settings.get_retrieval()["top_k"] == 11


def test_corrupt_file_falls_back_to_defaults():
    kb_settings.save("retrieval", kb_settings.DEFAULTS["retrieval"], user="admin")
    open(kb_settings.path(), "w", encoding="utf-8").write("{ not json")
    cfg, source = kb_settings.get_all()
    assert source == "default"
    assert cfg["chunking"] == kb_settings.DEFAULTS["chunking"]


def test_partial_section_merged_over_defaults():
    """文件里只写了 chunk_size 也不该让别的字段消失"""
    p = kb_settings.path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"chunking": {"chunk_size": 400}, "retrieval": {}}, open(p, "w", encoding="utf-8"))
    cfg, _ = kb_settings.get_all()
    assert cfg["chunking"]["chunk_size"] == 400
    assert cfg["chunking"]["overlap"] == kb_settings.DEFAULTS["chunking"]["overlap"]


def test_get_all_exposes_updated_metadata():
    """谁在什么时候改的 —— 必须能从 get_all 读出来, 且各分支形状一致"""
    cfg, _ = kb_settings.get_all()
    assert cfg["updated_at"] == "" and cfg["updated_by"] == ""

    kb_settings.save("retrieval", kb_settings.DEFAULTS["retrieval"], user="admin")
    cfg, _ = kb_settings.get_all()
    assert isinstance(cfg["updated_at"], str) and cfg["updated_at"]
    assert cfg["updated_by"] == "admin"


def test_callers_cannot_poison_the_cache():
    """调用方改返回值不该污染进程内缓存"""
    kb_settings.save("retrieval", {**kb_settings.DEFAULTS["retrieval"], "top_k": 3}, user="admin")
    cfg, _ = kb_settings.get_all()
    cfg["retrieval"]["top_k"] = 999
    cfg["chunking"]["chunk_size"] = 999
    cfg2, _ = kb_settings.get_all()
    assert cfg2["retrieval"]["top_k"] == 3
    assert cfg2["chunking"]["chunk_size"] == kb_settings.DEFAULTS["chunking"]["chunk_size"]
