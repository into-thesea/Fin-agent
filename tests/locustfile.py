"""
Fin-Agent Locust 性能测试 — 金融理财场景

模拟多类用户打对话主链路 (此前是文档问答时代的旧场景, 已作废):

  ChatUser        高频 — 产品咨询 / 收益口径 / 存款保险 / 费率
  ComplaintUser   中频 — 投诉与风险举报 (确定性转人工)
  MultiTurnUser   中频 — 多轮追问 (槽位填充 / 会话记忆)
  BackofficeUser  低频 — 健康检查 / 系统状态 / 监控

启动:
  locust -f tests/locustfile.py --host=http://localhost:8001
  无头: locust -f tests/locustfile.py --host=http://localhost:8001 \
        --headless -u 20 -r 2 -t 60s

注: 用 /chat/sync (非流式), locust 统计更干净; 要看 SSE 首字延迟用
    scripts/load_test.py 或 scripts/synthetic_users.py。
"""

import random

from locust import HttpUser, task, between

# 演示客户只有 3 个 (src/business/mock_finance.py PROFILES)
USERS = ["u001", "u002", "u003"]

PRODUCT_Q = [
    "稳盈添利30天业绩比较基准是多少",
    "有哪些 R2 的理财",
    "存款保险保不保理财产品",
    "银行理财是存款吗",
    "理财产品赎回到账要几天",
    "申购费和赎回费怎么收",
    "七日年化和万份收益有什么区别",
    "R3 的产品适合我吗",
]

COMPLAINT_Q = [
    "我要投诉你们的服务",
    "太差了，我要投诉",
    "有人打电话说能帮我操作理财，是不是诈骗",
    "我要举报飞单",
]

MULTI_TURN = [
    ["我想买理财", "稳盈添利30天", "5万"],
    ["有哪些R2产品", "那第一个的期限多久"],
    ["这只产品风险大吗", "和我现在持有的比呢"],
]

# 知识库外 → 制造空召回, 顺便压一下兜底话术
OOS_Q = [
    "你们银行的股价今天多少",
    "帮我推荐一只科技股",
    "今天的天气怎么样",
]


class _ChatBase(HttpUser):
    """公共: 每个虚拟用户独立 session, 避免全站共享一个会话导致缓存串答测不出来"""

    abstract = True
    wait_time = between(1, 3)

    def on_start(self):
        self.session_id = f"locust-{random.randint(100000, 999999)}"
        self.user_id = random.choice(USERS)

    def ask(self, query: str, name: str = "/chat/sync"):
        with self.client.post(
            "/api/v1/chat/sync",
            json={"query": query, "session_id": self.session_id, "user_id": self.user_id},
            name=name, timeout=240, catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "answer" not in resp.text:
                resp.failure("响应缺 answer 字段")


class ChatUser(_ChatBase):
    """高频 — 产品咨询"""

    weight = 5

    @task(10)
    def product_qa(self):
        self.ask(random.choice(PRODUCT_Q), name="/chat/sync [产品咨询]")

    @task(2)
    def out_of_scope(self):
        self.ask(random.choice(OOS_Q), name="/chat/sync [知识库外]")


class ComplaintUser(_ChatBase):
    """中频 — 投诉 / 举报 → 确定性转人工 + 建工单"""

    weight = 2
    wait_time = between(3, 8)

    @task
    def complaint(self):
        self.ask(random.choice(COMPLAINT_Q), name="/chat/sync [投诉/举报]")


class MultiTurnUser(_ChatBase):
    """中频 — 多轮追问, 压槽位填充与会话记忆"""

    weight = 2
    wait_time = between(2, 5)

    @task
    def convo(self):
        for i, q in enumerate(random.choice(MULTI_TURN)):
            self.ask(q, name=f"/chat/sync [多轮-第{i + 1}轮]")


class BackofficeUser(HttpUser):
    """低频 — 健康检查 (只打免鉴权端点)

    /api/v1/system/status 与 /api/v1/monitor/* 都挂在 require_roles 下,
    虚拟用户没 token 会 401 —— 要压这些接口得先登录拿 Bearer。
    """

    weight = 1
    wait_time = between(5, 15)

    @task(3)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def health_ready(self):
        self.client.get("/health/ready", name="/health/ready")
