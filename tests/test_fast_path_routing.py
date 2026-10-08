"""图拓扑: 过闸才快答, 不过闸升级."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.cs_graph import _STAGE_MSG, build_graph


def _edges():
    g = build_graph()
    return {(e.source, e.target) for e in g.get_graph().edges}


def test_simple_fact_reachable_only_via_triage():
    """simple_fact_node 只能由 route_retrieve_node 到达 (classify 不再直连)"""
    edges = _edges()
    assert ("memory_node", "simple_fact_node") not in edges
    assert ("route_retrieve_node", "simple_fact_node") in edges


def test_classify_only_splits_greeting():
    edges = _edges()
    assert ("memory_node", "greeting_node") in edges
    assert ("memory_node", "route_retrieve_node") in edges


def test_upgrade_edges_still_present():
    """不过闸时仍能落到完整管道"""
    edges = _edges()
    assert ("route_retrieve_node", "slot_check_node") in edges
    assert ("route_retrieve_node", "unknown_node") in edges


def test_stage_msg_covers_new_nodes():
    assert "route_retrieve_node" in _STAGE_MSG
    assert "simple_fact_node" in _STAGE_MSG


def test_route_next_three_way():
    """_route_next 必须按 fast_path 分岔, 而不是只看 intent"""
    import inspect

    from src.graph import cs_graph
    src = inspect.getsource(cs_graph.build_graph)
    assert 'state.get("fast_path")' in src, "_route_next 未按 fast_path 分岔"


if __name__ == "__main__":
    test_simple_fact_reachable_only_via_triage()
    test_classify_only_splits_greeting()
    test_upgrade_edges_still_present()
    test_stage_msg_covers_new_nodes()
    test_route_next_three_way()
    print("OK: fast path routing")
