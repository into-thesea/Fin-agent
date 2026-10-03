"""快速通道: router.route() 必须与检索/生成并发, 不得排在答案之后白等一个 LLM 往返.

覆盖 75% 流量, 串行时每次请求多花一个路由 LLM 往返 (实测约 0.5~1.6s)。
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.graph.cs_graph as G

SLEEP = 0.5  # 每个 LLM 往返的模拟耗时


class _SlowRouter:
    def route(self, query, dialog_context=""):
        time.sleep(SLEEP)
        return types.SimpleNamespace(
            intent=types.SimpleNamespace(value="simple_fact"), confidence=0.9)


class _SlowClient:
    def chat(self, messages):
        time.sleep(SLEEP)
        return "ans"

    def chat_stream(self, messages):
        time.sleep(SLEEP)
        yield "ans"


def test_route_overlaps_retrieval_and_generation():
    # router 是 build_graph 内的局部实例, 必须在建图前替换 RouterAgent 工厂
    saved = (G.retrieve_knowledge, G.RouterAgent, G.create_client)
    G.retrieve_knowledge = lambda q, k: (time.sleep(SLEEP), {
        "text": "KB", "sources": ["s"], "contexts": {}})[1]
    G.RouterAgent = lambda *a, **k: _SlowRouter()
    G.create_client = lambda cheap=False: _SlowClient()
    try:
        node = G.build_graph().nodes["simple_fact_node"].bound.func
        t0 = time.time()
        out = node({"query": "存款保险保多少"})
        elapsed = time.time() - t0
    finally:
        G.retrieve_knowledge, G.RouterAgent, G.create_client = saved

    # 检索→生成同线程串行 = 2*SLEEP; 路由并发后总耗时应仍接近 2*SLEEP, 而非 3*SLEEP
    assert elapsed < 2 * SLEEP + 0.3, f"route 未并发, 耗时 {elapsed:.2f}s (串行约 {3*SLEEP:.2f}s)"
    assert out["intent"] == "simple_fact" and out["answer"] == "ans"


if __name__ == "__main__":
    test_route_overlaps_retrieval_and_generation()
    print("OK: fast path runs route concurrently")
