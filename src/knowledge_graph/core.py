"""
知识图谱核心: 三元组加载 / 内存邻接图 / Neo4j 适配 (同接口)

设计: 图谱逻辑对后端透明。
  - InMemoryGraph: 从 data/finance_kb/kg_triples.jsonl 载入内存邻接 (零依赖, 离线可跑)
  - Neo4jGraph:   同 neighbors/path 接口, 用 neo4j driver 执行 Cypher (Docker 部署后启用)
GraphRetriever 优先 Neo4j (可用时), 否则内存降级 —— 链路不断。
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KG_PATH = os.path.join(PROJECT_ROOT, "data", "finance_kb", "kg_triples.jsonl")


def load_triples(path: str = KG_PATH) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logger.warning("知识图谱三元组读取失败 (空图): %s", e)
        return []


class InMemoryGraph:
    """内存邻接图 (有向). 每个三元组头节点出边存储 (relation, tail, doc, note)。"""

    def __init__(self, triples: list):
        self._out: dict = {}
        self._in: dict = {}
        for t in triples:
            h, rel, tail = t.get("head", ""), t.get("relation", ""), t.get("tail", "")
            if not h or not tail:
                continue
            self._out.setdefault(h, []).append((rel, tail, t.get("doc", ""), t.get("note", "")))
            self._in.setdefault(tail, []).append((rel, h, t.get("doc", ""), t.get("note", "")))

    @property
    def nodes(self) -> set:
        return set(self._out) | set(self._in)

    def neighbors(self, name: str, max_edges: int = 8) -> list:
        """出边邻居 → [(rel, tail, doc)] (含反向以 head 入边作为补充信息)"""
        out = [(rel, tail, doc, note, "out") for rel, tail, doc, note in self._out.get(name, [])]
        # 反向(被指向)关系也纳入, 覆盖"存款保险 -covered_by<- 存款"读法的缺口
        inbound = [(rel, h, doc, note, "in") for rel, h, doc, note in self._in.get(name, [])]
        return (out + inbound)[:max_edges]

    def path(self, start: str, end: str, max_hops: int = 3) -> Optional[list]:
        """有向 BFS 找 start→end 最短路径 (沿出边)。返回 [(head, rel, tail, doc, note), ...] 或 None"""
        if start == end:
            return []
        from collections import deque
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == end:
                break
            if len(prev) - 1 > max_hops:
                break
            for rel, tail, doc, note, _ in self._iter_out(cur):
                if tail not in prev:
                    prev[tail] = (cur, rel, doc, note)
                    q.append(tail)
        if end not in prev:
            return None
        # 回溯还原路径
        steps = []
        cur = end
        while prev[cur] is not None:
            parent, rel, doc, note = prev[cur]
            steps.append((parent, rel, cur, doc, note))
            cur = parent
        steps.reverse()
        return steps

    def _iter_out(self, name: str):
        return [(rel, tail, doc, note, "out") for rel, tail, doc, note in self._out.get(name, [])]


# ── Neo4j 同接口适配 (Docker 部署后可用; 不可用时返回 None 供上层降级内存) ──

class Neo4jGraph:
    """Neo4j 后端: neighbors / path 与 InMemoryGraph 同语义, 用 Cypher 查询。

    环境变量: NEO4J_URI(默认 bolt://127.0.0.1:7687) NEO4J_USER(neo4j)
              NEO4J_PASSWORD(默认 kefu2026neo, 与 docker-compose 一致)
    """

    def __init__(self):
        self._driver = None
        self._uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self._user = os.getenv("NEO4J_USER", "neo4j")
        self._password = os.getenv("NEO4J_PASSWORD", "kefu2026neo")
        self._checked = False

    def _connect(self):
        if self._checked:
            return self._driver
        self._checked = True
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self._driver.verify_connectivity()
            logger.info("Neo4j 连接成功: %s", self._uri)
        except Exception as e:
            self._driver = None
            logger.info("Neo4j 不可用 (%s), 图谱走内存降级", e)
        return self._driver

    @property
    def available(self) -> bool:
        return self._connect() is not None

    def _run(self, cypher: str, **params) -> list:
        d = self._connect()
        if d is None:
            return []
        try:
            with d.session() as s:
                return list(s.run(cypher, **params))
        except Exception as e:
            logger.warning("Neo4j 查询失败: %s", e)
            return []

    def neighbors(self, name: str, max_edges: int = 8) -> list:
        rows = self._run(
            "MATCH (n {name:$name})-[r]-(m) "
            "RETURN type(r) AS rel, m.name AS tail, coalesce(r.doc,'') AS doc, coalesce(r.note,'') AS note "
            "LIMIT $max_edges", name=name, max_edges=max_edges)
        return [(r["rel"], r["tail"], r["doc"], r["note"], "out") for r in rows]

    def path(self, start: str, end: str, max_hops: int = 3) -> Optional[list]:
        # Neo4j 的 shortestPath 变量长度 `[*1..N]` 不支持参数, 必须内联字面量
        hops = max(1, int(max_hops))
        rows = self._run(
            "MATCH p = shortestPath((a {name:$start})-[*1..%d]->(b {name:$end})) "
            "UNWIND relationships(p) AS r "
            "RETURN startNode(r).name AS h, type(r) AS rel, endNode(r).name AS t, "
            "coalesce(r.doc,'') AS doc, coalesce(r.note,'') AS note" % hops,
            start=start, end=end)
        if not rows:
            return None
        return [(r["h"], r["rel"], r["t"], r["doc"], r["note"]) for r in rows]


_GRAPH_CACHE = None


def get_graph() -> InMemoryGraph:
    """全局内存图 (小图常驻)"""
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = InMemoryGraph(load_triples())
    return _GRAPH_CACHE


def triple_to_sentence(h: str, rel: str, t: str, doc: str = "", note: str = "") -> str:
    rel_zh = {
        "is_type_of": "属于", "is_subtype_of": "属于",
        "risk_level": "风险等级为", "deposit_insured": "受存款保险保障为",
        "is_not": "不是", "not_covered_by": "不受", "covered_by": "受",
        "coverage_limit": "赔付限额为", "applies_to": "仅适用于", "excludes": "不保障",
        "requires": "要求", "has_risk": "存在", "is": "为", "has": "设有",
        "allows": "允许", "feature": "特点是", "must_not": "禁止",
        "must_include": "必须包含", "recorded_in": "记入", "affects": "影响",
        "sold_as": "作为", "subject_to": "受", "suitability_demo": "适当性示例",
        "coverage_scope": "覆盖",
    }.get(rel, f"[{rel}]")
    tail_txt = f"{t}（{note}）" if note else t
    s = f"{h} {rel_zh} {tail_txt}"
    if doc:
        s += f"（依据 {doc}）"
    return s
