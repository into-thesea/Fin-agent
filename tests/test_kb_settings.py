import json, os, sys
import inspect

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


def test_save_twice_in_a_row():
    """连存两次必须都成功 —— 第二次读到的配置里已有 updated_at, 曾经的
    dict 推导式会在这上面抛 ValueError, 表现为"第一次保存成功, 之后每次都 500"。"""
    kb_settings.save("retrieval", {**kb_settings.DEFAULTS["retrieval"], "top_k": 9}, user="admin")
    kb_settings.save("retrieval", {**kb_settings.DEFAULTS["retrieval"], "top_k": 7}, user="admin")
    assert kb_settings.get_retrieval()["top_k"] == 7
    assert kb_settings.get_retrieval()["rerank_pool"] == kb_settings.DEFAULTS["retrieval"]["rerank_pool"]


def test_callers_cannot_poison_the_cache():
    """调用方改返回值不该污染进程内缓存"""
    kb_settings.save("retrieval", {**kb_settings.DEFAULTS["retrieval"], "top_k": 3}, user="admin")
    cfg, _ = kb_settings.get_all()
    cfg["retrieval"]["top_k"] = 999
    cfg["chunking"]["chunk_size"] = 999
    cfg2, _ = kb_settings.get_all()
    assert cfg2["retrieval"]["top_k"] == 3
    assert cfg2["chunking"]["chunk_size"] == kb_settings.DEFAULTS["chunking"]["chunk_size"]


def test_pipeline_passes_configured_chunk_params():
    """切分参数必须真的传到 smart_chunk_pdf —— 否则页面上改了没有任何作用"""
    from src.core import pipeline_manager
    src = inspect.getsource(pipeline_manager.ingest_document)
    assert "get_chunking" in src
    assert "chunk_size=" in src and "overlap=" in src
    assert "max_chunk_content=" in src


def _header_pages():
    """两节长文本 —— 章节检测到手, 且每节都长到需要细切"""
    return [{"page": 1, "text": "一、产品要素\n" + "甲" * 1000 + "\n二、收益口径\n" + "乙" * 1000}]


def test_max_chunk_content_is_a_real_parameter():
    """content_size 上限必须可传入 —— 它原先只是模块常量, 页面上改了不起作用"""
    from src.core.smart_chunk import _chaptered_chunk, MAX_CHUNK_CONTENT

    pages = _header_pages()
    # 默认(不传) == 显式传常量值, 保证默认行为不变
    assert _chaptered_chunk(pages, "x.pdf", 350, 70) == \
        _chaptered_chunk(pages, "x.pdf", 350, 70, max_chunk_content=MAX_CHUNK_CONTENT)

    wide = _chaptered_chunk(pages, "x.pdf", 350, 70, max_chunk_content=300)
    narrow = _chaptered_chunk(pages, "x.pdf", 350, 70, max_chunk_content=80)
    assert max(len(c["content"]) for c in narrow) < max(len(c["content"]) for c in wide)
    assert len(narrow) > len(wide)
