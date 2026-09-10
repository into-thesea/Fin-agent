"""
AgenticAgent — Function Calling 自主决策引擎

让模型通过工具循环自主决定调用哪些工具 (retrieve_knowledge / web_search / query_graph)，
再基于工具结果生成带来源的最终回答。

回退保护:
  - 工具循环异常/失败 → 回退到"预取检索 + 普通 chat"(旧 domain agent 行为), 保证不中断服务
  - 未知工具 → 返回提示, 不崩溃

流式:
  - answer_stream: 先跑工具循环收集上下文(发 stage 事件), 再对最终答案流式输出
"""

import json
import logging
from typing import Optional, Iterator, Tuple, Any

from src.llm.llm_client import create_client, append_tool_result
from src.agents.prompts import TOOL_GUIDE
from src.tools.registry import TOOL_DEFINITIONS, EXECUTE_TOOLS

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 6


class AgenticAgent:
    """基于 function calling 的工具自主决策 Agent"""

    def __init__(self, system_prompt: str, intent: str = "general",
                 client: Any = None):
        """
        Args:
            system_prompt: 意图角色 prompt (已含边界块)
            intent: 意图名 (用于日志/审计)
            client: 可注入的 LLM 客户端 (便于测试)
        """
        self.system_prompt = system_prompt
        self.intent = intent
        self.client = client or create_client()

    # ── 主入口 ─────────────────────────────────────

    def answer(self, query: str, prefetched_contexts: Optional[dict] = None,
               **kwargs) -> dict:
        """工具循环作答, 失败回退到预取检索 + 普通 chat"""
        try:
            return self._answer_with_tools(query)
        except Exception as e:
            logger.warning("Agentic[%s] 工具循环失败, 回退到普通 chat: %s",
                           self.intent, e)
            return self._answer_fallback(query, prefetched_contexts)

    def answer_stream(self, query: str, prefetched_contexts: Optional[dict] = None,
                      on_event=None) -> Iterator[Tuple[Any, bool]]:
        """
        流式版本: 工具循环(非流式)收集上下文 → 最终答案流式输出
        yield (token|result, done)
        """
        gathered = None
        try:
            gathered = self._gather_context(query, on_event=on_event)
        except Exception as e:
            logger.warning("Agentic[%s] 流式工具循环失败, 回退: %s", self.intent, e)
            gathered = self._fallback_gather(query, prefetched_contexts)

        ctx_text = gathered.get("ctx_text", "")
        sources = gathered.get("sources", [])

        user_input = f"问题: {query}\n\n【检索到的参考信息】:\n{ctx_text}\n\n请基于上述信息作答, 并标注来源。"
        full_answer = []
        for content in self.client.chat_stream(
            messages=[
                {"role": "system", "content": self.system_prompt + "\n\n" + TOOL_GUIDE},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        ):
            full_answer.append(content)
            yield content, False

        result = {
            "answer": "".join(full_answer),
            "sources": sources,
            "contexts": gathered.get("contexts", {"local": [], "global": [], "graph": []}),
            "handoff": gathered.get("handoff", False),
        }
        yield result, True

    # ── 工具循环 ─────────────────────────────────────

    def _answer_with_tools(self, query: str) -> dict:
        """完整工具循环, 直到模型返回文本"""
        messages = [{"role": "system", "content": self.system_prompt + "\n\n" + TOOL_GUIDE},
                    {"role": "user", "content": query}]
        sources, contexts = [], {"local": [], "global": [], "graph": []}
        handoff = False  # 工具返回 handoff=True 时置位 (如超退换期退款)

        for turn in range(MAX_TOOL_TURNS):
            result = self.client.chat_with_tools(messages, TOOL_DEFINITIONS)
            if result.get("type") != "tool_calls":
                return {"answer": result.get("text", ""),
                        "sources": _dedupe(sources), "contexts": contexts,
                        "handoff": handoff}

            # 先追加 assistant tool_calls 消息, 再追加各 tool 结果 (OpenAI 顺序要求)
            assistant_msg = {"role": "assistant", "content": None, "tool_calls": []}
            for call in result.get("calls", []):
                assistant_msg["tool_calls"].append({
                    "id": call.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                    },
                    "_raw_function_call": call.get("_raw_function_call"),
                })
            messages.append(assistant_msg)

            for call in result.get("calls", []):
                out = self._execute_tool(call)
                append_tool_result(messages, call, out["text"])
                sources.extend(out["sources"])
                handoff = handoff or bool(out.get("handoff"))
                ctx = out.get("contexts")
                if ctx:
                    contexts["local"].extend(ctx.get("local", []))
                    contexts["graph"].extend(ctx.get("graph", []))
                    if ctx.get("global"):
                        contexts["global"] = ctx.get("global")
            logger.info("Agentic[%s] 工具循环第 %d 轮: 调用 %s",
                        self.intent, turn + 1,
                        [c.get("name") for c in result.get("calls", [])])

        logger.warning("Agentic[%s] 工具循环达到上限 %d, 返回占位", self.intent, MAX_TOOL_TURNS)
        return {"answer": "工具调用次数过多，请简化问题后重试。",
                "sources": _dedupe(sources), "contexts": contexts,
                "handoff": handoff}

    def _gather_context(self, query: str, on_event=None) -> dict:
        """仅收集工具上下文 (不生成最终答案), 供流式模式使用"""
        messages = [{"role": "system", "content": self.system_prompt + "\n\n" + TOOL_GUIDE},
                    {"role": "user", "content": query}]
        ctx_texts, sources, contexts = [], [], {"local": [], "global": [], "graph": []}
        handoff = False

        for turn in range(MAX_TOOL_TURNS):
            result = self.client.chat_with_tools(messages, TOOL_DEFINITIONS)
            if result.get("type") != "tool_calls":
                break  # 模型直接给文本 → 无工具可收集

            assistant_msg = {"role": "assistant", "content": None, "tool_calls": []}
            for call in result.get("calls", []):
                assistant_msg["tool_calls"].append({
                    "id": call.get("id", ""), "type": "function",
                    "function": {"name": call.get("name", ""),
                                 "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)},
                    "_raw_function_call": call.get("_raw_function_call"),
                })
            messages.append(assistant_msg)

            for call in result.get("calls", []):
                if on_event:
                    on_event({"type": "stage", "stage": "tool_call",
                              "message": f"🔧 调用工具 {call.get('name')}..."})
                out = self._execute_tool(call)
                append_tool_result(messages, call, out["text"])
                ctx_texts.append(out["text"])
                sources.extend(out["sources"])
                handoff = handoff or bool(out.get("handoff"))
                ctx = out.get("contexts")
                if ctx:
                    contexts["local"].extend(ctx.get("local", []))
                    contexts["graph"].extend(ctx.get("graph", []))
                    if ctx.get("global"):
                        contexts["global"] = ctx.get("global")

        return {"ctx_text": "\n\n".join(ctx_texts), "sources": _dedupe(sources),
                "contexts": contexts, "handoff": handoff}

    def _execute_tool(self, call: dict) -> dict:
        """执行单个工具调用"""
        name = call.get("name", "")
        args = call.get("arguments") or {}
        handler = EXECUTE_TOOLS.get(name)
        if not handler:
            logger.warning("Agentic[%s] 未知工具: %s", self.intent, name)
            return {"text": f"未知工具: {name}", "sources": [], "contexts": None}
        try:
            return handler(**args)
        except Exception as e:
            logger.warning("Agentic[%s] 工具 %s 执行失败: %s", self.intent, name, e)
            return {"text": f"工具 {name} 执行失败: {str(e)[:100]}", "sources": [], "contexts": None}

    # ── 回退路径 ─────────────────────────────────────

    def _answer_fallback(self, query: str, prefetched_contexts: Optional[dict]) -> dict:
        """预取检索 + 普通 chat (与旧 domain agent 行为一致)"""
        # 一律走 retrieve_knowledge: 统一实体对齐抑制 (知识库外的实体不喂其他公司内容)
        # 不能用 prefetched_contexts 直接格式化, 否则绕过抑制 → 问华为答比亚迪
        from src.tools.registry import retrieve_knowledge
        kr = retrieve_knowledge(query, 5)
        user_input = f"问题: {query}\n\n{kr['text']}\n\n请基于上述信息作答, 并标注来源。"
        answer = self.client.chat(
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": user_input}],
            temperature=0.1,
        )
        return {"answer": answer, "sources": kr["sources"], "contexts": kr.get("contexts") or {}}

    def _fallback_gather(self, query: str, prefetched_contexts: Optional[dict]) -> dict:
        """回退路径的上下文收集"""
        # 一律走 retrieve_knowledge (保证实体对齐抑制生效)
        from src.tools.registry import retrieve_knowledge
        kr = retrieve_knowledge(query, 5)
        return {"ctx_text": kr["text"], "sources": kr["sources"],
                "contexts": kr.get("contexts") or {}}


# ── 模块级辅助 ─────────────────────────────────────

_retriever = None


def _default_retriever():
    """惰性单例 HybridRetriever"""
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


def _dedupe(items: list) -> list:
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out
