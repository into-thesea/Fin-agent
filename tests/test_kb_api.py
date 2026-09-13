"""知识库管理台接口测试 —— 策略读写 (/api/v1/kb/strategy) + 片段浏览 (/api/v1/kb/chunks)

覆盖四件事:
  1. 整组 admin-only (匿名 401)
  2. GET 回显「当前值 / 代码默认值 / 哪些字段偏离默认」
  3. PUT 的字段级 400 + 生效范围回显 (retrieval=下次请求 / chunking=下次上传)
  4. 分块分页 + 来源/小节/关键词筛选, 以及翻页越界/空结果不外抛
"""
import json
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.routes.auth import create_token
from src.core import kb_settings
from src.infra.paths import CHUNKS_PROCESSED_PATH  # noqa: F401  (测试里被 monkeypatch 的路径常量)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_headers():
    return {"Authorization": f"Bearer {create_token('admin', role='admin')}"}


@pytest.fixture(autouse=True)
def temp_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(kb_settings, "path", lambda: str(tmp_path / "kb_settings.json"))
    kb_settings.invalidate_cache()
    yield
    kb_settings.invalidate_cache()


def test_anonymous_rejected(client):
    assert client.get("/api/v1/kb/strategy").status_code == 401


def test_get_returns_current_and_default(client, admin_headers):
    r = client.get("/api/v1/kb/strategy", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["chunking"]["current"]["chunk_size"] == 350
    assert body["chunking"]["default"]["chunk_size"] == 350
    assert body["chunking"]["changed"] == []
    assert body["source"] == "default"


def test_put_retrieval_takes_effect_immediately(client, admin_headers):
    r = client.put("/api/v1/kb/strategy", headers=admin_headers,
                   json={"section": "retrieval", "values": {"top_k": 9, "rerank_pool": 20}})
    assert r.status_code == 200
    body = r.json()
    assert body["applies_to"] == "next_request"
    assert "top_k" in body["changed"]
    assert kb_settings.get_retrieval()["top_k"] == 9


def test_put_chunking_reports_next_upload(client, admin_headers):
    r = client.put("/api/v1/kb/strategy", headers=admin_headers,
                   json={"section": "chunking",
                         "values": {"chunk_size": 400, "overlap": 70, "max_chunk_content": 300}})
    assert r.status_code == 200
    # 切分参数只影响之后新上传的 PDF —— 不是"重建后生效"
    assert r.json()["applies_to"] == "next_upload"


def test_put_invalid_returns_field_level_detail(client, admin_headers):
    r = client.put("/api/v1/kb/strategy", headers=admin_headers,
                   json={"section": "chunking",
                         "values": {"chunk_size": 300, "overlap": 300, "max_chunk_content": 300}})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["field"] == "overlap"
    assert "chunk_size" in detail["rule"]


# --- 片段查询 (/api/v1/kb/chunks) ---

@pytest.fixture
def fake_chunks(tmp_path, monkeypatch):
    """造 5 条分块, 覆盖两个来源"""
    p = tmp_path / "chunks_processed.jsonl"
    rows = [
        {"chunk_id": f"c{i}", "source": "a.md" if i < 3 else "b.md",
         "section": "产品要素" if i % 2 == 0 else "收益口径",
         "page": 1, "content": f"内容{i} 稳健理财", "content_hash": f"h{i}"}
        for i in range(5)
    ]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # 两处都要 patch: 路由模块在模块级 import 了这个常量
    monkeypatch.setattr("src.infra.paths.CHUNKS_PROCESSED_PATH", str(p))
    monkeypatch.setattr("src.api.routes.kb_console.CHUNKS_PROCESSED_PATH", str(p))
    return rows


def test_chunks_pagination(client, admin_headers, fake_chunks):
    r = client.get("/api/v1/kb/chunks?page=1&size=2", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5 and len(body["items"]) == 2
    assert body["items"][0]["chunk_id"] == "c0"
    assert body["items"][0]["length"] == len("内容0 稳健理财")


def test_chunks_filter_by_source(client, admin_headers, fake_chunks):
    body = client.get("/api/v1/kb/chunks?source=b.md", headers=admin_headers).json()
    assert body["total"] == 2
    assert {i["source"] for i in body["items"]} == {"b.md"}


def test_chunks_filter_by_keyword(client, admin_headers, fake_chunks):
    body = client.get("/api/v1/kb/chunks?q=内容3", headers=admin_headers).json()
    assert body["total"] == 1 and body["items"][0]["chunk_id"] == "c3"


def test_chunks_size_capped(client, admin_headers, fake_chunks):
    """size 上限 100 —— 不设上限等于允许一次拉全库"""
    assert client.get("/api/v1/kb/chunks?size=9999", headers=admin_headers).status_code == 422


def test_chunks_anonymous_rejected(client):
    assert client.get("/api/v1/kb/chunks").status_code == 401


# --- 边界: 翻页翻到越界不能报错, 空结果 ≠ 知识库坏了 ---

def test_chunks_page_zero_rejected(client, admin_headers, fake_chunks):
    assert client.get("/api/v1/kb/chunks?page=0", headers=admin_headers).status_code == 422


def test_chunks_page_beyond_end_is_empty_not_error(client, admin_headers, fake_chunks):
    r = client.get("/api/v1/kb/chunks?page=999&size=20", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5 and body["items"] == []


def test_chunks_filter_no_match_is_empty(client, admin_headers, fake_chunks):
    body = client.get("/api/v1/kb/chunks?q=不存在的关键词", headers=admin_headers).json()
    assert body["total"] == 0 and body["items"] == []


def test_chunks_missing_file_returns_empty(client, admin_headers, tmp_path, monkeypatch):
    """分块文件还没生成时返回空列表, 不是 500"""
    missing = str(tmp_path / "nope.jsonl")
    monkeypatch.setattr("src.api.routes.kb_console.CHUNKS_PROCESSED_PATH", missing)
    r = client.get("/api/v1/kb/chunks", headers=admin_headers)
    assert r.status_code == 200 and r.json()["total"] == 0


# --- 全量重建 (/api/v1/kb/rebuild) ---

def test_rebuild_requires_worker(client, admin_headers, monkeypatch):
    """没 worker 时明确 503, 不能默默投一个没人消费的任务"""
    monkeypatch.setattr("src.api.routes.kb_console.is_celery_worker_running", lambda: False)
    r = client.post("/api/v1/kb/rebuild", headers=admin_headers)
    assert r.status_code == 503
    assert "Worker" in r.json()["detail"]


def test_rebuild_submits_task_and_returns_id(client, admin_headers, monkeypatch):
    """有 worker 时投递任务并回 task_id —— 进度复用已有的 /api/v1/tasks/{id}"""
    import src.tasks.kb_tasks as kb_tasks
    from src.db.manager import DBManager

    monkeypatch.setattr("src.api.routes.kb_console.is_celery_worker_running", lambda: True)
    monkeypatch.setattr(kb_tasks, "rebuild_kb_task",
                        type("T", (), {"delay": staticmethod(lambda: type("R", (), {"id": "task-1"})())}))
    recorded = {}
    monkeypatch.setattr(DBManager, "create_task",
                        lambda self, task_id, filename: recorded.update(task_id=task_id, filename=filename))

    r = client.post("/api/v1/kb/rebuild", headers=admin_headers)
    assert r.status_code == 200
    assert r.json() == {"task_id": "task-1", "status": "accepted"}
    assert recorded["task_id"] == "task-1"


def test_rebuild_anonymous_rejected(client):
    assert client.post("/api/v1/kb/rebuild").status_code == 401
