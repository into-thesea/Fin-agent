"""知识库管理台接口测试 —— 策略读写 (/api/v1/kb/strategy) + 片段浏览 (/api/v1/kb/chunks)

覆盖四件事:
  1. 整组 admin-only (匿名 401)
  2. GET 回显「当前值 / 代码默认值 / 哪些字段偏离默认」
  3. PUT 的字段级 400 + 生效范围回显 (retrieval=下次请求 / chunking=下次上传)
  4. 分块分页 + 来源/小节/关键词筛选, 以及翻页越界/空结果不外抛
"""
import json
import os
import sys

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


def test_put_enabling_rerank_without_api_key_rejected(client, admin_headers, monkeypatch):
    """勾一个框就能让每条问答都检索失败 —— 必须在保存时拦下, 而不是运行时优雅降级"""
    from src.config import settings
    monkeypatch.setattr(settings, "dashscope_api_key", "")
    r = client.put("/api/v1/kb/strategy", headers=admin_headers,
                   json={"section": "retrieval", "values": {"rerank_enabled": True}})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["field"] == "rerank_enabled" and "DASHSCOPE_API_KEY" in detail["rule"]


def test_put_enabling_rerank_with_api_key_ok(client, admin_headers, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "dashscope_api_key", "test-key")
    r = client.put("/api/v1/kb/strategy", headers=admin_headers,
                   json={"section": "retrieval", "values": {"rerank_enabled": True}})
    assert r.status_code == 200
    assert kb_settings.get_retrieval()["rerank_enabled"] is True


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


# --- 检索调试 (/api/v1/kb/retrieval-test) ---

def _fake_debug(**over):
    """一份最小的 retrieve_debug 返回; 默认三路都有, 单通路两条"""
    out = {
        "dense": [{"chunk_id": "c1", "source": "a.md", "rank": 1, "score": 0.9}],
        "sparse": [{"chunk_id": "c2", "source": "b.md", "rank": 1, "score": 3.1}],
        "fused": [
            {"chunk_id": "both", "source": "a.md", "section": "产品要素", "rank": 1,
             "rrf_score": 0.03, "dense_rank": 1, "sparse_rank": 1,
             "rerank_score": None, "found_by": ["dense", "sparse"]},
            {"chunk_id": "only_dense", "source": "a.md", "section": "收益口径", "rank": 2,
             "rrf_score": 0.01, "dense_rank": 2, "sparse_rank": None,
             "rerank_score": None, "found_by": ["dense"]},
            {"chunk_id": "only_sparse", "source": "b.md", "section": "费率", "rank": 3,
             "rrf_score": 0.01, "dense_rank": None, "sparse_rank": 2,
             "rerank_score": None, "found_by": ["sparse"]},
        ],
        "graph": {"entries": [], "entities": []},
        "config_used": {"top_k": 5},
        "elapsed_ms": 1.0,
    }
    out.update(over)
    return out


def test_retrieval_test_filters_single_path(client, admin_headers, monkeypatch):
    """only_single_path 只留"靠单条通路召回"的项 —— 调试时最值得看的就是这些"""
    from src.retrieval.retriever import HybridRetriever

    monkeypatch.setattr(HybridRetriever, "retrieve_debug",
                        lambda self, q, top_k=5: _fake_debug(dense=[], sparse=[]))
    r = client.post("/api/v1/kb/retrieval-test", headers=admin_headers,
                    json={"query": "稳盈添利30天风险等级", "top_k": 5, "only_single_path": True})
    assert r.status_code == 200
    ids = [i["chunk_id"] for i in r.json()["fused"]]
    assert ids == ["only_dense", "only_sparse"]
    assert r.json()["single_path_count"] == 2


def test_retrieval_test_rejects_empty_query(client, admin_headers):
    r = client.post("/api/v1/kb/retrieval-test", headers=admin_headers,
                    json={"query": "   ", "top_k": 5})
    assert r.status_code == 400


def test_retrieval_test_rejects_top_k_out_of_range(client, admin_headers):
    """越界 top_k 会直接把参数透给 Milvus —— 在入口拦下, 别让它变成 500"""
    for bad in (0, 21):
        r = client.post("/api/v1/kb/retrieval-test", headers=admin_headers,
                        json={"query": "稳盈添利", "top_k": bad})
        assert r.status_code == 400, bad


def test_retrieval_test_reports_sparse_empty(client, admin_headers, monkeypatch):
    """稀疏路为空时 retrieve_debug 会退化成 dense[:top_k], found_by 全读成 ["dense"],
    会被误读成"关键词检索没命中" —— 调试台必须能区分这两件事"""
    from src.retrieval.retriever import HybridRetriever

    monkeypatch.setattr(HybridRetriever, "retrieve_debug",
                        lambda self, q, top_k=5: _fake_debug(sparse=[]))
    body = client.post("/api/v1/kb/retrieval-test", headers=admin_headers,
                       json={"query": "稳盈添利", "top_k": 5}).json()
    assert body["sparse_empty"] is True

    monkeypatch.setattr(HybridRetriever, "retrieve_debug",
                        lambda self, q, top_k=5: _fake_debug())
    body = client.post("/api/v1/kb/retrieval-test", headers=admin_headers,
                       json={"query": "稳盈添利", "top_k": 5}).json()
    assert body["sparse_empty"] is False


def test_retrieval_test_keeps_graph_error(client, admin_headers, monkeypatch):
    """图谱挂了要如实显示, 不能吞掉 error 键假装图谱没命中"""
    from src.retrieval.retriever import HybridRetriever

    monkeypatch.setattr(HybridRetriever, "retrieve_debug",
                        lambda self, q, top_k=5: _fake_debug(
                            graph={"entries": [], "entities": [], "error": "Neo4j 连接失败"}))
    body = client.post("/api/v1/kb/retrieval-test", headers=admin_headers,
                       json={"query": "稳盈添利", "top_k": 5}).json()
    assert body["graph"]["error"] == "Neo4j 连接失败"


def test_retrieval_test_failure_returns_504(client, admin_headers, monkeypatch):
    """Milvus 挂了要报 504 并带出原因 —— 调试台显示"检索执行失败, 原因是 X", 不是空结果"""
    from src.retrieval.retriever import HybridRetriever

    def boom(self, q, top_k=5):
        raise RuntimeError("Milvus 不可用")

    monkeypatch.setattr(HybridRetriever, "retrieve_debug", boom)
    r = client.post("/api/v1/kb/retrieval-test", headers=admin_headers,
                    json={"query": "稳盈添利", "top_k": 5})
    assert r.status_code == 504
    assert "Milvus 不可用" in r.json()["detail"]


def test_retrieval_test_anonymous_rejected(client):
    assert client.post("/api/v1/kb/retrieval-test",
                       json={"query": "稳盈添利"}).status_code == 401
