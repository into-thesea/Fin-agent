"""
TrendAgent — 趋势分析智能体

专注时间序列分析、增长率计算、周期性识别。
需要从图谱中获取同一指标在不同时间段的数据点。
"""

import os
import json
import logging

from dotenv import load_dotenv
from src.retrieval.retriever import HybridRetriever, format_graph_context
from src.llm.llm_client import create_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

from src.agents.prompts import TREND_SYSTEM_PROMPT


class TrendAgent:
    """趋势分析 Agent"""

    def __init__(self):
        self.client = create_client()
        self.retriever = HybridRetriever()

    def answer(self, query: str, entity_hint: str = None, context: dict = None,
                web_context: str = None, prefetched_contexts: dict = None) -> dict:
        """
        回答趋势分析问题

        Args:
            query: 用户问题
            entity_hint: 核心实体
            context: 可选的上下文
            web_context: 可选的 Web 搜索结果补充

        Returns:
            {"answer": str, "sources": list, "data_points": list, "trend_direction": str}
        """
        # 使用预取上下文 (避免重复检索)
        contexts = prefetched_contexts or self.retriever.hybrid_retrieve(query, entity_hint=entity_hint)

        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} [来源: {c.get('source', '未知')}]"
             for c in contexts['local']]
        )
        global_text = "\n".join(
            [f"- 宏观背景: {s}" for s in contexts['global']]
        )
        graph_text = format_graph_context(contexts.get('graph', []))
        web_text = f"\n\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""问题: {query}

【参考原文】:
{local_text}

【宏观背景】:
{global_text}

【图谱关系】:
{graph_text}{web_text}

请提供详细的趋势分析。"""

        answer = self.client.chat(
            messages=[
                {"role": "system", "content": TREND_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )
        sources = list(set(c['source'] for c in contexts['local']))

        # 简单判断趋势方向
        trend_direction = "unknown"
        trend_keywords = {
            "增长": "up",
            "上升": "up",
            "下降": "down",
            "下滑": "down",
            "持平": "stable",
            "波动": "volatile",
        }
        for kw, direction in trend_keywords.items():
            if kw in answer[:200]:
                trend_direction = direction
                break

        return {
            "answer": answer,
            "sources": sources,
            "trend_direction": trend_direction,
            "data_points": [],
            "contexts": contexts,  # 传给 ReviewerAgent，避免重复检索
        }

    # ── 流式版本 ─────────────────────────────────────

    def answer_stream(self, query: str, entity_hint: str = None,
                      context: dict = None, web_context: str = None,
                      prefetched_contexts: dict = None):
        """流式版本, 逐 token 产出 (token, done) 二元组"""
        contexts = prefetched_contexts or self.retriever.hybrid_retrieve(query, entity_hint=entity_hint)

        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} [来源: {c.get('source', '未知')}]"
             for c in contexts['local']]
        )
        global_text = "\n".join(
            [f"- 宏观背景: {s}" for s in contexts['global']]
        )
        graph_text = format_graph_context(contexts.get('graph', []))
        web_text = f"\n\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""问题: {query}

【参考原文】:
{local_text}

【宏观背景】:
{global_text}

【图谱关系】:
{graph_text}{web_text}

请提供详细的趋势分析。"""

        full_answer = []
        sources = list(set(c['source'] for c in contexts['local']))

        for content in self.client.chat_stream(
            messages=[
                {"role": "system", "content": TREND_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        ):
            full_answer.append(content)
            yield content, False

        result = {
            "answer": "".join(full_answer),
            "sources": sources,
            "trend_direction": "unknown",
            "data_points": [],
            "contexts": contexts,
        }
        yield result, True
