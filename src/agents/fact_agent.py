"""
FactAgent — 事实查询智能体

专注回答具体数值/事实类问题，强调精确引用和溯源。
"""

import logging
import os

from dotenv import load_dotenv

from src.llm.llm_client import create_client
from src.retrieval.retriever import HybridRetriever, format_graph_context

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from src.agents.prompts import FACT_SYSTEM_PROMPT


class FactAgent:
    """事实查询 Agent"""

    def __init__(self):
        self.client = create_client()
        self.retriever = HybridRetriever()

    def answer(self, query: str, entity_hint: str = None, context: dict = None,
                web_context: str = None, prefetched_contexts: dict = None) -> dict:
        """
        回答事实类问题

        Args:
            query: 用户问题
            entity_hint: 核心实体 (来自 RouterAgent)
            context: 可选的外部上下文 (来自 RouterAgent 的实体/时间范围)
            web_context: 可选的 Web 搜索结果补充
            prefetched_contexts: 来自 AnalystAgent 预取的三路召回结果 (避免重复检索)

        Returns:
            {"answer": str, "sources": list, "confidence": str, "citations": list}
        """
        # 使用预取上下文 (避免重复检索 BGE+BM25+Neo4j)
        contexts = prefetched_contexts or self.retriever.hybrid_retrieve(query, entity_hint=entity_hint)

        local_text = "\n".join(
            [f"- (score:{c.get('score', 0):.3f}) {c.get('content', '')[:1500]} [来源: {c.get('source', '未知')}]"
             for c in contexts['local']]
        )
        graph_text = format_graph_context(contexts.get('graph', []))
        web_text = f"\n\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""问题: {query}

【参考原文】:
{local_text}

【知识图谱证据】:
{graph_text}{web_text}

请基于上述信息给出精确事实回答。"""

        answer = self.client.chat(
            messages=[
                {"role": "system", "content": FACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.05,
        )
        sources = list(set(c['source'] for c in contexts['local']))

        logger.info("FactAgent 完成: %s -> %d chars, %d sources",
                     query[:40], len(answer), len(sources))

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "high",
            "citations": sources,
            "contexts": contexts,  # 传给 ReviewerAgent，避免重复检索
        }

    # ── 流式版本 ─────────────────────────────────────

    def answer_stream(self, query: str, entity_hint: str = None,
                      context: dict = None, web_context: str = None,
                      prefetched_contexts: dict = None):
        """流式版本, 逐 token 产出 (token, done) 二元组"""
        contexts = prefetched_contexts or self.retriever.hybrid_retrieve(query, entity_hint=entity_hint)

        local_text = "\n".join(
            [f"- (score:{c.get('score', 0):.3f}) {c.get('content', '')[:1500]} [来源: {c.get('source', '未知')}]"
             for c in contexts['local']]
        )
        graph_text = format_graph_context(contexts.get('graph', []))
        web_text = f"\n\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""问题: {query}

【参考原文】:
{local_text}

【知识图谱证据】:
{graph_text}{web_text}

请基于上述信息给出精确事实回答。"""

        full_answer = []
        sources = list(set(c['source'] for c in contexts['local']))

        for content in self.client.chat_stream(
            messages=[
                {"role": "system", "content": FACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.05,
        ):
            full_answer.append(content)
            yield content, False

        # 最后一个包: yield 结果
        result = {
            "answer": "".join(full_answer),
            "sources": sources,
            "confidence": "high",
            "citations": sources,
            "contexts": contexts,
        }
        yield result, True
