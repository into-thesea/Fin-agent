"""
CompareAgent — 对比分析智能体

专注跨公司、跨指标的对比分析，需要同时检索多个实体的数据并结构化对比。
"""

import os
import json
import logging

from dotenv import load_dotenv
from src.retrieval.retriever import HybridRetriever
from src.llm.llm_client import create_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from src.agents.prompts import COMPARE_SYSTEM_PROMPT

class CompareAgent:
    """对比分析 Agent"""

    def __init__(self):
        self.client = create_client()
        self.retriever = HybridRetriever()

        # 对比关键词 (用于从查询中提取多个实体)
        self.compare_markers = ["vs", "对比", "比较", " versus ", " vs ",
                                "和", "与", "跟", "较"]

    def _extract_entities_from_query(self, query: str) -> list:
        """从对比查询中提取多个公司名称 (启发式)"""
        # 后续可以用 LLM 做更精确的提取
        entities = []
        # 先用简单分割尝试
        for marker in [" vs ", " VS ", " vs ", "和", "与", "、"]:
            if marker in query:
                parts = query.split(marker)
                # 提取疑似公司名 (首字母大写的词)
                for part in parts[:2]:
                    words = part.strip().split()
                    for w in words:
                        if w[0].isupper() and len(w) > 1:
                            entities.append(w)
                            break
                break
        return entities

    def answer(self, query: str, entity_hint: str = None, context: dict = None,
                web_context: str = None, prefetched_contexts: dict = None) -> dict:
        """
        回答对比分析问题

        Args:
            query: 用户问题
            entity_hint: 核心实体 (RouterAgent 提供)
            context: 可选的上下文
            web_context: 可选的 Web 搜索结果补充

        Returns:
            {"answer": str, "sources": list, "entities_compared": list}
        """
        # 提取对比实体
        if context and context.get("entities"):
            compare_entities = [e["name"] for e in context["entities"]
                               if e.get("type") == "Company"]
        else:
            compare_entities = self._extract_entities_from_query(query)

        logger.info("对比实体: %s", compare_entities)

        # 为每个实体分别检索，收集更全面的上下文
        all_local = []
        all_graph = []
        all_global = []
        seen_sources = set()

        for ent in compare_entities[:3]:  # 最多对比 3 个
            ctx = self.retriever.hybrid_retrieve(query, entity_hint=ent)
            all_local.extend(ctx['local'])
            all_graph.extend(ctx['graph'])
            all_global = ctx.get('global', [])  # 全局摘要相同，取最后一次

        # 去重
        unique_local = []
        for c in all_local:
            key = c.get('chunk_id', c['content'][:50])
            if key not in seen_sources:
                seen_sources.add(key)
                unique_local.append(c)

        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} [来源: {c.get('source', '未知')}]"
             for c in unique_local[:10]]
        )
        graph_text = "\n".join(all_graph[:15])
        web_text = f"\n\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""问题: {query}
对比对象: {', '.join(compare_entities) if compare_entities else entity_hint}

【双方参考原文】:
{local_text}

【双方图谱关系】:
{graph_text}{web_text}

请提供详细的结构化对比分析。"""

        answer = self.client.chat(
            messages=[
                {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )
        sources = list(set(c['source'] for c in unique_local))

        return {
            "answer": answer,
            "sources": sources,
            "entities_compared": compare_entities,
            "contexts": {"local": unique_local[:10], "graph": all_graph[:15], "global": all_global},
        }

    # ── 流式版本 ─────────────────────────────────────

    def answer_stream(self, query: str, entity_hint: str = None,
                      context: dict = None, web_context: str = None,
                      prefetched_contexts: dict = None):
        """流式版本, 逐 token 产出 (token, done) 二元组"""
        if context and context.get("entities"):
            compare_entities = [e["name"] for e in context["entities"]
                               if e.get("type") == "Company"]
        else:
            compare_entities = self._extract_entities_from_query(query)

        all_local = []
        all_graph = []
        all_global = []
        seen_sources = set()

        for ent in compare_entities[:3]:
            ctx = self.retriever.hybrid_retrieve(query, entity_hint=ent)
            all_local.extend(ctx['local'])
            all_graph.extend(ctx['graph'])
            all_global = ctx.get('global', [])

        unique_local = []
        for c in all_local:
            key = c.get('chunk_id', c['content'][:50])
            if key not in seen_sources:
                seen_sources.add(key)
                unique_local.append(c)

        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} [来源: {c.get('source', '未知')}]"
             for c in unique_local[:10]]
        )
        graph_text = "\n".join(all_graph[:15])
        web_text = f"\n\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""问题: {query}
对比对象: {', '.join(compare_entities) if compare_entities else entity_hint}

【双方参考原文】:
{local_text}

【双方图谱关系】:
{graph_text}{web_text}

请提供详细的结构化对比分析。"""

        full_answer = []
        sources = list(set(c['source'] for c in unique_local))

        for content in self.client.chat_stream(
            messages=[
                {"role": "system", "content": COMPARE_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        ):
            full_answer.append(content)
            yield content, False

        result = {
            "answer": "".join(full_answer),
            "sources": sources,
            "entities_compared": compare_entities,
            "contexts": {"local": unique_local[:10], "graph": all_graph[:15], "global": all_global},
        }
        yield result, True
