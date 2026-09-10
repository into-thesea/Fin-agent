"""
Fin-Agent Locust 性能测试

模拟 100 并发用户执行:
  - 健康检查 (高频)
  - 知识库查询 (中频)
  - 文档上传 (低频)
  - 图谱查询 (中频)

启动:
  locust -f tests/locustfile.py --host=http://localhost:8001
  # 或 Docker (旧 compose 部署, api 在 8000):
  # docker run -p 8089:8089 -v $PWD:/mnt locustio/locust \
  #   -f /mnt/tests/locustfile.py --host=http://api:8000
"""

import random
from locust import HttpUser, task, between


class HealthCheckUser(HttpUser):
    """高频 — 健康检查"""
    wait_time = between(1, 3)

    @task(10)
    def health_check(self):
        self.client.get("/health")

    @task(3)
    def health_ready(self):
        self.client.get("/health/ready")


class KnowledgeUser(HttpUser):
    """中频 — 知识库查询"""
    wait_time = between(3, 8)

    @task(5)
    def list_documents(self):
        self.client.get("/api/v1/knowledge/documents")

    @task(3)
    def get_stats(self):
        self.client.get("/api/v1/knowledge/stats")

    @task(2)
    def get_metrics(self):
        self.client.get("/api/v1/monitor/metrics")


class GraphQueryUser(HttpUser):
    """中频 — 图谱查询"""
    wait_time = between(5, 15)

    entities = [
        "比亚迪股份有限公司",
        "宁德时代新能源科技股份有限公司",
        "Tesla, Inc.",
        "小米集团",
        "NIO Inc.",
    ]

    @task(4)
    def query_entity(self):
        entity = random.choice(self.entities)
        self.client.get(f"/api/v1/graph/entity/{entity}")

    @task(1)
    def query_entity_2hop(self):
        entity = random.choice(self.entities)
        self.client.get(f"/api/v1/graph/entity/{entity}?hops=2")


class UploadSimulationUser(HttpUser):
    """低频 — 模拟上传 (只测端点可用性，不上传真实文件)"""
    wait_time = between(30, 60)

    @task(1)
    def upload_small(self):
        # 只测端点是否接受请求，不上传大文件
        self.client.post(
            "/api/v1/knowledge/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 small test", "application/pdf")},
            timeout=30,
        )
