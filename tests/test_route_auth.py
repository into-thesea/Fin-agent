"""
路由鉴权回归测试

固化一条通用不变量: **管理类路由必须整组挂 require_roles, 健康检查保持公开。**

背景: src/api/routes/monitor.py 曾经整组没有守卫 (knowledge.py / handoff.py 都有),
于是审计日志(含用户原始提问)、问题复盘明细、以及 badcases 的写接口
全部匿名可读可写 —— 实测无 token 也返回 200。

这类漏洞靠人肉 review 容易漏 (一个路由文件忘加 dependencies 就中招),
所以按「整组」而不是「逐个端点」来断言: 新增端点默认继承守卫, 不会漏。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.auth import create_token


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin_token():
    return create_token("admin", role="admin")


# 必须鉴权的路由组 (前缀 → 说明)
PROTECTED_GROUPS = {
    "/api/v1/monitor": "监控与审计日志 (含用户原始提问)",
    "/api/v1/badcases": "问题复盘明细与写接口",
    "/api/v1/knowledge": "知识库读写",
    "/api/v1/handoff": "转人工工单",
    "/api/v1/eval": "评测数据",
    "/api/v1/kb": "知识库管理台 (策略配置读写)",
}

# 允许匿名访问的路由 (必须保持公开)
PUBLIC_PATHS = ["/health", "/health/ready"]

# 各组里至少探测一个真实存在的端点 —— 只看前缀会在路由被改名时静默通过
PROBE_PATHS = [
    "/api/v1/monitor/metrics",
    "/api/v1/monitor/audit-logs",
    "/api/v1/monitor/sli?window=1h",
    "/api/v1/system/status",
    "/api/v1/badcases",
    "/api/v1/eval/summary",
    "/api/v1/knowledge/documents",
    "/api/v1/knowledge/stats",
    "/api/v1/handoff/queue",
    "/api/v1/kb/strategy",
]


@pytest.mark.parametrize("path", PROBE_PATHS)
def test_protected_routes_reject_anonymous(client, path):
    """匿名请求必须被拒 (401), 不能返回数据。"""
    r = client.get(path)
    assert r.status_code == 401, f"{path} 匿名可访问 (status={r.status_code})"


@pytest.mark.parametrize("path", PROBE_PATHS)
def test_protected_routes_allow_admin(client, admin_token, path):
    """带合法 token 必须放行 —— 否则守卫挂错地方, 前端会全挂。"""
    r = client.get(path, headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code != 401, f"{path} 合法 token 被拒 (status={r.status_code})"


@pytest.mark.parametrize("path", PUBLIC_PATHS)
def test_public_routes_stay_open(client, path):
    """健康检查必须保持公开 —— 挂上守卫会让 K8s/负载均衡探针全部失败。"""
    assert client.get(path).status_code == 200


def test_each_protected_group_has_a_probe():
    """每个受保护路由组都要有被探测的端点。

    新增一组受保护路由却忘了加探针时, 这条会失败 —— 否则那组等于没测。
    """
    for prefix in PROTECTED_GROUPS:
        assert any(p.startswith(prefix) for p in PROBE_PATHS), f"{prefix} 没有探针端点"
