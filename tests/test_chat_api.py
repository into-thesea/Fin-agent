"""FastAPI 路由测试 — 健康 / 认证 / 反馈 (TestClient, 不依赖外部服务)"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import settings
from src.api.routes.auth import create_token, decode_token


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── token 签发/解析 (纯函数) ─────────────────────

def test_token_roundtrip():
    t = create_token("u001", role="user")
    d = decode_token(t)
    assert d["user_id"] == "u001"
    assert d["role"] == "user"


def test_token_tamper_rejected():
    t = create_token("u001")
    forged = t[:-3] + "abc"
    assert decode_token(forged) is None
    assert decode_token("") is None


# ── 路由 ─────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


def test_login_when_not_configured(client):
    if settings.admin_password:
        pytest.skip("admin_password 已配置, 走登录成功分支")
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 403


def test_auth_flow_when_configured(client):
    if not settings.admin_password:
        pytest.skip("admin_password 未配置")
    r = client.post("/api/v1/auth/login", json={
        "username": settings.admin_username, "password": settings.admin_password,
    })
    assert r.status_code == 200
    token = r.json()["token"]
    p = client.get("/api/v1/auth/profile", headers={"Authorization": f"Bearer {token}"})
    assert p.status_code == 200
    assert p.json()["username"] == settings.admin_username


def test_feedback_rating_validation(client):
    r = client.post("/api/v1/chat/feedback", json={"session_id": "rt_1", "rating": 5})
    assert r.status_code == 400


def test_chat_empty_query_rejected(client):
    r = client.post("/api/v1/chat/sync", json={"query": ""})
    assert r.status_code == 400
