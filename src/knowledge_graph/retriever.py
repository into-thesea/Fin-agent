"""
GraphRetriever — 知识图谱多跳检索 (GraphRAG 局部查询思路)

流程: 实体解析(产品名/概念别名) → 双实体路径查询(有向BFS ≤3跳) → 剩余实体邻居补全
      → 每条带 doc 溯源转自然语言。
后端: Neo4j 可用走 Neo4jGraph, 否则内存邻接降级 (链路不断)。

返回条目与向量/BM25 语境同构: {"content": str, "source": "知识图谱-<doc>", "path": [节点...]}。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from src.core.slot_filler import KNOWN_PRODUCTS  # 产品全名 (catalog.jsonl 单一事实源)
from .core import get_graph, Neo4jGraph, triple_to_sentence

# 概念枢纽: 出边多且泛化, 不作为单实体邻居展开的入口
_HUBS = {"理财产品", "存款", "公募基金", "保险产品"}

# 关系类/多跳语义信号: 命中才把图谱证据注入 prompt (推荐/罗列类单实体查询不注入)
RELATIONAL_HINTS = (
    "是存款吗", "是不是存款", "保不保", "保障", "影响", "属于", "区别",
    "和", "与", "能退", "退保", "犹豫期", "是什么", "还是",
)

# 产品口语别名 (与 tools/registry 对齐; 单独维护避免 import 环)
_PRODUCT_ALIASES = {
    "大额存单3年期": ["大额存单"],
    "结构性存款·挂钩黄金3个月": ["结构性存款"],
    "稳盈添利30天": ["稳盈添利"],
    "安心固收90天": ["安心固收"],
    "平衡增利180天": ["平衡增利"],
    "私银聚享混合策略": ["私银聚享"],
    "日日盈现金管理类": ["日日盈"],
    "安鑫纯债基金": ["安鑫纯债"],
    "远见成长混合基金": ["混合基金"],
    "盛世稳赢增额终身寿": ["增额终身寿", "盛世稳赢"],
    "安享颐年养老年金": ["养老年金", "安享颐年"],
}

# 概念/政策节点别名
_CONCEPT_ALIASES = {
    "理财产品": ["理财产品", "银行理财", "理财", "银行理财产品"],
    "存款": ["存款", "存款类产品", "存款产品"],
    "存款保险": ["存款保险"],
    "公募基金": ["公募基金", "基金"],
    "保险产品": ["保险产品", "保险"],
    "结构性存款": ["结构性存款"],
    "增额终身寿险": ["增额终身寿险"],
    "养老年金保险": ["养老年金保险"],
    "个人征信": ["个人征信", "征信"],
    "房贷审批": ["房贷审批", "房贷"],
    "信用卡逾期": ["信用卡逾期", "逾期记录", "逾期"],
    "适当性匹配": ["适当性匹配", "适当性"],
    "业绩比较基准": ["业绩比较基准", "比较基准"],
    "七日年化": ["七日年化", "万份收益"],
    "双录": ["双录", "录音录像"],
    "犹豫期15天": ["犹豫期", "犹豫期15天"],
}


def _build_alias_map() -> dict:
    """term(别名/全名) → 规范节点名。按别名长度降序供最长优先匹配。"""
    m = {}
    for full in KNOWN_PRODUCTS:
        m[full] = full
        for a in _PRODUCT_ALIASES.get(full, []):
            m[a] = full
    for canon, terms in _CONCEPT_ALIASES.items():
        for t in terms:
            m[t] = canon
    return m


class GraphRetriever:
    def __init__(self, backend: Optional[str] = None):
        # backend: "neo4j" | "memory" | None(自动: neo4j 可用则用)
        self._aliases = sorted(_build_alias_map().items(), key=lambda kv: -len(kv[0]))
        self._use_neo4j = backend == "neo4j" or (backend is None and Neo4jGraph().available)

    def resolve(self, query: str) -> list:
        """返回查询命中的规范节点名 (最长优先, 去重, ≤3)"""
        found, used = [], set()
        for term, canon in self._aliases:
            if canon in used:
                continue
            if term in query:
                found.append(canon)
                used.add(canon)
            if len(found) >= 3:
                break
        return found

    def retrieve(self, query: str, max_hops: int = 3) -> dict:
        """图谱检索 → {"entries": [ {content, source, path} ], "entities": [..]}"""
        entities = self.resolve(query)
        if not entities:
            return {"entries": [], "entities": entities}
        store = self._backend_graph()
        seen, entries = set(), []
        paths: list = []

        def add_triple(h, rel, t, doc, note):
            key = (h, rel, t)
            if key in seen:
                return
            seen.add(key)
            content = triple_to_sentence(h, rel, t, doc, note)
            entries.append({
                "content": content,
                "source": f"知识图谱-{doc}" if doc else "知识图谱",
                "path": [h, t],
            })

        # 1) 双实体有向路径 (取前两个实体的最短链, 反向再试一次)
        if len(entities) >= 2:
            for a, b in [(entities[0], entities[1]), (entities[1], entities[0])]:
                chain = store.path(a, b, max_hops)
                if chain:
                    paths = [(x[0], x[1], x[2]) for x in chain]
                    for h, rel, t, doc, note in chain:
                        add_triple(h, rel, t, doc, note)
                    break
        # 2) 单实体邻居补全 (含未进入路径的其余实体)
        # 概念枢纽节点(理财/存款/基金/保险)出边多且泛化, 展开噪音大 → 跳过, 由路径/定向边覆盖
        for ent in entities:
            if ent in _HUBS:
                continue
            for rel, tail, doc, note, _ in store.neighbors(ent):
                add_triple(ent, rel, tail, doc, note)

        # 数量上限, 保持注入上下文精简
        return {"entries": entries[:8], "entities": entities, "paths": paths}

    def _backend_graph(self):
        return Neo4jGraph() if self._use_neo4j else get_graph()
