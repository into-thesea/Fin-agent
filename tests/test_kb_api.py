"""知识库管理台接口测试 —— 策略读写 (/api/v1/kb/strategy)

覆盖三件事:
  1. 整组 admin-only (匿名 401)
  2. GET 回显「当前值 / 代码默认值 / 哪些字段偏离默认」
  3. PUT 的字段级 400 + 生效范围回显 (retrieval=下次请求 / chunking=下次上传)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.routes.auth import create_token
from src.core import kb_settings


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
