"""灌库脚本的收敛语义: 源里没有的边/节点必须从 Neo4j 清掉

只 MERGE 不删的话, 源删掉一条边它就永久留在图里 —— 实测 `存款保险 applies_to
定期存款` 就是这么攒下来的, 攒到对账时才发现 Neo4j 比源多 1 条旧边、少 24 条新边。
这段逻辑是 Cypher 谓词, 脱离 Neo4j 验不了, 所以库里没起时整个文件跳过。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from src.knowledge_graph.core import KG_PATH, load_triples


@pytest.fixture
def driver():
    neo4j = pytest.importorskip("neo4j")
    d = neo4j.GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "kefu2026neo")),
    )
    try:
        d.verify_connectivity()
    except Exception as e:
        d.close()
        pytest.skip(f"Neo4j 未启动: {type(e).__name__}")
    yield d
    d.close()


def test_loader_converges_graph_to_source(driver):
    with driver.session() as s:
        s.run("MERGE (a:Entity {name:'__conv_h'}) MERGE (b:Entity {name:'__conv_t'}) "
              "MERGE (a)-[r:REL {name:'__conv_rel'}]->(b)")

    import load_kg_to_neo4j
    load_kg_to_neo4j.main()

    triples = load_triples(KG_PATH)
    with driver.session() as s:
        probe = s.run("MATCH (n:Entity) WHERE n.name STARTS WITH '__conv' "
                      "RETURN count(n) AS c").single()["c"]
        n_rels = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
        n_nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]

    assert probe == 0, "源里没有的探针节点没被清掉 —— 收敛没生效"
    assert n_rels == len(triples), f"关系数 {n_rels} != 源 {len(triples)} 条"
    assert n_nodes == len({t[k] for t in triples for k in ("head", "tail")})
