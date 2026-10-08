"""triage 的路由与检索必须并发 —— 快答依赖"意图+置信度是免费拿到的"这个前提.

覆盖 75% 流量。串行时每次请求多花一个路由 LLM 往返 (实测约 0.5~1.6s)。
(原先这段守卫挂在 simple_fact_node 上; 五道闸改造后快答节点不再自己检索,
 该职责整体移到了 route_retrieve_node。)
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.graph.cs_graph as G
import src.retrieval.retriever as R

SLEEP = 0.5  # 每个 LLM 往返 / 检索的模拟耗时


class _SlowRouter:
    def route(self, query, dialog_context=""):
        time.sleep(SLEEP)
        return types.SimpleNamespace(
            intent=types.SimpleNamespace(value="product_consult"),
            confidence=0.95, entities=[], explanation="")

    def strong_signal_rule(self, query):
        return None, None


class _SlowRetriever:
    def __init__(self, *a, **k):
        pass

    def hybrid_retrieve(self, query, entity_hint=None, top_k=None):
        time.sleep(SLEEP)
        return {"local": [{"score": 0.80, "content": "稳盈添利30天 业绩比较基准 3.1%",
                           "source": "prod_p002.md"}],
                "sources": ["prod_p002.md"]}


def test_triage_runs_route_and_retrieval_concurrently():
    saved = (G.RouterAgent, G.decompose_query, R.HybridRetriever)
    G.RouterAgent = lambda *a, **k: _SlowRouter()
    G.decompose_query = lambda q: [q]          # 单问题快速路径, 不额外调用
    R.HybridRetriever = _SlowRetriever
    try:
        node = G.build_graph().nodes["route_retrieve_node"].bound.func
        t0 = time.time()
        out = node({"query": "稳盈添利30天的风险等级是多少"})
        elapsed = time.time() - t0
    finally:
        G.RouterAgent, G.decompose_query, R.HybridRetriever = saved

    # 串行 = 2*SLEEP; 并发后应接近 SLEEP
    assert elapsed < SLEEP * 1.5, f"路由与检索未并发, 耗时 {elapsed:.2f}s (串行约 {2*SLEEP:.2f}s)"
    assert out["intent"] == "product_consult"
    assert out["kb_text"], "triage 必须产出 kb_text"
    assert out["fast_path"] is True, "该样本应过闸"


def test_simple_fact_node_no_longer_retrieves():
    """快答节点复用 triage 的 kb_text, 不得自己再检索一次"""
    import inspect
    src = inspect.getsource(G.build_graph)
    node_src = src.split("def simple_fact_node")[1].split("def _build_dialog_context")[0]
    assert "retrieve_knowledge" not in node_src
    assert "router.route" not in node_src
    assert 'state.get("kb_text")' in node_src


if __name__ == "__main__":
    test_triage_runs_route_and_retrieval_concurrently()
    test_simple_fact_node_no_longer_retrieves()
    print("OK: triage runs route concurrently")
