"""三元组人工审核测试 —— 通过 / 拒绝 / 重复跳过

关键回归点: approve 必须写"源"文件 kg_domain_triples.jsonl。
kg_triples.jsonl 是 build_finance_kb 的产物 (见 tests/test_sync_kb.py 对事实源的断言),
直接追加会被下次构建整个覆盖, 审核结果静默丢失。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import kb_console
from src.api.routes.auth import create_token

# 导入时的取值, 供下面的护栏测试使用 (fixture 会把模块属性 monkeypatch 到 tmp)
_TARGET_AT_IMPORT = kb_console.KG_TRIPLES_PATH


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_headers():
    return {"Authorization": f"Bearer {create_token('admin', role='admin')}"}


@pytest.fixture(autouse=True)
def temp_paths(tmp_path, monkeypatch):
    """把待审目录与目标文件都挪到 tmp, 不碰真实知识库"""
    pending = tmp_path / "kg_pending"
    pending.mkdir()
    target = tmp_path / "kg_domain_triples.jsonl"
    target.write_text(
        json.dumps({"head": "存款保险", "relation": "is", "tail": "保障制度",
                    "doc": "deposit_insurance.md"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kb_console, "PENDING_DIR", pending)
    monkeypatch.setattr(kb_console, "KG_TRIPLES_PATH", target)
    yield pending, target


def _seed(pending, *triples):
    for t in triples:
        t.setdefault("status", "pending")
        t.setdefault("confidence", 1.0)
        entry = dict(t)
        entry["id"] = kb_console._triple_id(t)
        with open(pending / "deposit_insurance.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_anonymous_rejected(client):
    assert client.get("/api/v1/kb/triples/pending").status_code == 401


def test_audit_targets_source_not_artifact():
    """护栏: 审核写入目标必须是手写源, 不是 build_finance_kb 的产物.

    写到 kg_triples.jsonl 时, 下次 render_kg_triples() 会整个覆盖该文件,
    审核通过的三元组静默丢失。
    """
    assert _TARGET_AT_IMPORT.name == "kg_domain_triples.jsonl"


def test_pending_list(client, admin_headers, temp_paths):
    pending, _ = temp_paths
    _seed(pending, {"head": "存款保险", "relation": "applies_to", "tail": "活期存款",
                    "doc": "deposit_insurance.md", "source_text": "存款保险保障活期存款"})
    r = client.get("/api/v1/kb/triples/pending", headers=admin_headers)
    assert r.status_code == 200
    assert len(r.json()["items"]) == 1


def test_approve_writes_source_not_artifact(client, admin_headers, temp_paths):
    pending, target = temp_paths
    _seed(pending, {"head": "存款保险", "relation": "applies_to", "tail": "定期存款",
                    "doc": "deposit_insurance.md", "source_text": "含定期存款"})
    tid = kb_console._triple_id({"head": "存款保险", "relation": "applies_to", "tail": "定期存款"})

    r = client.post("/api/v1/kb/triples/audit",
                    json={"id": tid, "action": "approve"}, headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "approved"

    rows = _lines(target)
    assert any(t["tail"] == "定期存款" for t in rows), "审核通过的三元组未写入源文件"
    # 待审条目状态已流转
    entry = _lines(pending / "deposit_insurance.jsonl")[0]
    assert entry["status"] == "approved"


def test_reject_marks_without_writing(client, admin_headers, temp_paths):
    pending, target = temp_paths
    _seed(pending, {"head": "存款保险", "relation": "is", "tail": "理财产品",
                    "doc": "deposit_insurance.md", "source_text": "错误抽取"})
    tid = kb_console._triple_id({"head": "存款保险", "relation": "is", "tail": "理财产品"})
    before = len(_lines(target))

    r = client.post("/api/v1/kb/triples/audit",
                    json={"id": tid, "action": "reject"}, headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"
    assert len(_lines(target)) == before, "拒绝的三元组不该写入"
    assert _lines(pending / "deposit_insurance.jsonl")[0]["status"] == "rejected"


def test_duplicate_skipped(client, admin_headers, temp_paths):
    pending, target = temp_paths
    _seed(pending, {"head": "存款保险", "relation": "is", "tail": "保障制度",
                    "doc": "deposit_insurance.md", "source_text": "与源文件重复"})
    tid = kb_console._triple_id({"head": "存款保险", "relation": "is", "tail": "保障制度"})
    before = len(_lines(target))

    r = client.post("/api/v1/kb/triples/audit",
                    json={"id": tid, "action": "approve"}, headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "duplicate"
    assert len(_lines(target)) == before, "重复三元组不该二次写入"


def test_unknown_id_and_bad_action(client, admin_headers, temp_paths):
    assert client.post("/api/v1/kb/triples/audit",
                       json={"id": "deadbeef", "action": "approve"},
                       headers=admin_headers).status_code == 404
    assert client.post("/api/v1/kb/triples/audit",
                       json={"id": "deadbeef", "action": "maybe"},
                       headers=admin_headers).status_code == 400
