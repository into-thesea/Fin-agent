"""
GraphRetriever — 知识图谱多跳检索 (GraphRAG 局部查询思路)

流程: 实体解析(产品名/概念别名) → 双实体路径查询(有向BFS ≤3跳) → 剩余实体邻居补全
      → 每条带 doc 溯源转自然语言。
后端: 运行时只走 Neo4jGraph; Neo4j 不可用直接报错, 不降级内存图。

返回条目与向量/BM25 语境同构: {"content": str, "source": "知识图谱-<doc>", "path": [节点...]}。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from src.core.slot_filler import KNOWN_PRODUCTS  # 产品全名 (catalog.jsonl 单一事实源)

from .core import Neo4jGraph, get_graph, triple_to_sentence

# 概念枢纽: 出边多且泛化, 不作为单实体邻居展开的入口
_HUBS = {"理财产品", "存款", "公募基金", "保险产品"}

# 关系类/多跳语义信号: 命中才把图谱证据注入 prompt (推荐/罗列类单实体查询不注入)
RELATIONAL_HINTS = (
    "是存款吗", "是不是存款", "保不保", "保障", "影响", "属于", "区别",
    "和", "与", "能退", "退保", "犹豫期", "是什么", "还是",
)

# 产品口语别名 (与 tools/registry 对齐; 单独维护避免 import 环)
# 只保留指向具体产品的简称/俗称, 不用泛化品类词(如"国债"/"货币基金"/"混合基金")
# —— 品类词会同时匹配多款产品, 查询扩展时误替换反而缩小召回
_PRODUCT_ALIASES = {
    "日日盈现金管理类": ["日日盈", "日日盈理财"],
    "稳盈添利30天": ["稳盈添利", "稳盈30天", "添利30天"],
    "安心固收90天": ["安心固收", "安心90天", "固收90天"],
    "平衡增利180天": ["平衡增利", "平衡180天", "增利180天"],
    "私银聚享混合策略": ["私银聚享", "聚享混合", "私银理财"],
    "大额存单3年期": ["大额存单", "3年大额存单", "大额存单3年", "三年期大额存单"],
    "结构性存款·挂钩黄金3个月": ["结构性存款", "黄金结构性存款", "3个月结构性存款"],
    "安鑫纯债基金": ["安鑫纯债", "安鑫基金"],
    "远见成长混合基金": ["远见成长", "远见混合", "成长混合", "远见成长基金"],
    "盛世稳赢增额终身寿": ["增额终身寿", "盛世稳赢", "增额寿", "盛世稳赢增额寿"],
    "安享颐年养老年金": ["养老年金", "安享颐年", "安享养老"],
    "天天利货币基金": ["天天利", "天天利货币"],
    "沪深300指数增强基金": ["沪深300", "指数增强", "300指数基金", "沪深300基金"],
    "全球精选QDII股票基金": ["QDII", "全球精选", "QDII基金"],
    "稳健配置FOF基金": ["FOF", "稳健配置", "FOF基金", "稳健FOF"],
    "康健无忧重大疾病保险": ["重疾险", "康健无忧", "康健无忧重疾"],
    "百万医疗保险": ["百万医疗", "百万医疗险", "百万医疗保"],
    "综合意外伤害保险": ["意外险", "综合意外"],
    "家庭支柱定期寿险": ["定寿", "家庭支柱", "家庭定寿"],
    "储蓄国债(电子式)3年期": ["储蓄国债", "电子式国债", "3年期国债", "储蓄式国债"],
    "记账式国债10年期": ["记账式国债", "10年期国债", "十年国债", "记账国债"],
    "可转债(可转换公司债券)": ["可转债", "可转换债券", "转债", "可转换公司债"],
    "固定收益类集合信托计划": ["集合信托", "固定收益信托"],
    "账户黄金(积存金)": ["账户黄金", "积存金", "黄金积存"],
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
        """backend: "neo4j"(默认运行时后端) | "memory"(仅单测显式指定)

        运行时不再降级到内存图: 内存图与 Neo4j 的节点/边可能不一致,
        悄悄换掉数据源等于悄悄改变答案的证据依据。
        """
        self._aliases = sorted(_build_alias_map().items(), key=lambda kv: -len(kv[0]))
        if backend == "memory":
            self._graph = get_graph()
            return
        neo = Neo4jGraph()
        if not neo.available:
            raise RuntimeError(
                "Neo4j 不可用 —— 知识图谱检索无法执行 (不再降级到内存图)。"
                "请启动 Neo4j (docker compose up -d neo4j)。"
            )
        self._graph = neo

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
        return self._graph
