"""
把 data/finance_kb/kg_triples.jsonl 灌入 Neo4j (Docker 部署的完整图谱后端)

幂等: 为 (name) 建唯一约束后按 MERGE 写入节点与关系。
用法:
  1) 起 Docker: docker compose up -d neo4j
  2) 灌库:      .venv/Scripts/python.exe scripts/load_kg_to_neo4j.py
  3) 校验:      见脚本末尾统计
"""

from __future__ import annotations

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("load_kg_to_neo4j")

from src.knowledge_graph.core import KG_PATH, load_triples


def main():
    triples = load_triples(KG_PATH)
    if not triples:
        logger.error("无三元组可导入: %s", KG_PATH)
        sys.exit(1)

    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "kefu2026neo")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except Exception as e:
        logger.error("无法连接 Neo4j (%s): %s\n请先: docker compose up -d neo4j", uri, e)
        sys.exit(1)

    with driver.session() as s:
        s.run("CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
        for t in triples:
            h, rel, tail = t["head"], t["relation"], t["tail"]
            s.run(
                "MERGE (a:Entity {name:$h}) MERGE (b:Entity {name:$t}) "
                "MERGE (a)-[r:REL {name:$rel}]->(b) "
                "SET r.doc = $doc, r.note = $note",
                h=h, t=tail, rel=rel, doc=t.get("doc", ""), note=t.get("note", ""),
            )
        n_nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        n_rels = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
    driver.close()
    logger.info("Neo4j 灌库完成: %d 节点 / %d 关系 (源: %s)", n_nodes, n_rels, KG_PATH)


if __name__ == "__main__":
    main()
