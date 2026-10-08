"""知识图谱多跳检索测试 (内存后端, 离线确定性)"""

from src.knowledge_graph.core import load_triples
from src.knowledge_graph.retriever import GraphRetriever


def _retriever() -> GraphRetriever:
    return GraphRetriever(backend="memory")


def test_triples_loaded():
    triples = load_triples()
    assert len(triples) >= 60


def test_resolve_entities():
    r = _retriever()
    ents = r.resolve("银行理财是存款吗 存款保险赔吗")
    assert "理财产品" in ents and "存款" in ents and "存款保险" in ents
    # 产品简称映射到目录全名
    assert r.resolve("大额存单 有存款保险吗")[0] == "大额存单3年期"


def test_multi_hop_deposit_insurance():
    r = _retriever()
    res = r.retrieve("大额存单受存款保险保障吗")
    assert res["entities"]
    content = "\n".join(e["content"] for e in res["entries"])
    # 多跳链: 大额存单 属于 存款 → 存款保险 覆盖 存款
    assert "大额存单3年期" in content and "存款" in content and "存款保险" in content
    for e in res["entries"]:
        assert e["source"].startswith("知识图谱")


def test_credit_path():
    """三实体因果链: 信用卡逾期 → 个人征信 → 房贷审批"""
    r = _retriever()
    res = r.retrieve("信用卡逾期会影响房贷吗")
    content = "\n".join(e["content"] for e in res["entries"])
    assert "个人征信" in content and "房贷审批" in content


def test_hub_not_expanded_for_list_query():
    """罗列类查询(单概念枢纽)不产生图谱噪音条目"""
    r = _retriever()
    res = r.retrieve("有哪些理财产品推荐")
    assert res["entries"] == []
