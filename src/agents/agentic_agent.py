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
import concurrent.futures
from typing import Optional, Iterator, Tuple, Any

from src.llm.llm_client import create_client, append_tool_result
from src.agents.prompts import TOOL_GUIDE
from src.tools.registry import TOOL_DEFINITIONS, EXECUTE_TOOLS

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 6

# 补检索后追加给模型的约束: 只许用刚喂进去的原文, 没有的就说没有
_GROUNDING_REMINDER = ("请只依据以上原文作答；原文没有的内容直接说"
                       "\"知识库中未找到该信息\"，不要标注未出现在上面的来源。")


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
        """完整工具循环, 直到模型返回文本.
        优化: (1) 重复调用相同工具且已有上下文时提前终止; (2) 多工具并行执行.
        """
        messages = [{"role": "system", "content": self.system_prompt + "\n\n" + TOOL_GUIDE},
                    {"role": "user", "content": query}]
        sources, contexts = [], {"local": [], "global": [], "graph": []}
        handoff = False
        called_tool = False
        called_tool_names = set()  # 跟踪已调用过的工具名, 用于提前终止判断

        for turn in range(MAX_TOOL_TURNS):
            result = self.client.chat_with_tools(messages, TOOL_DEFINITIONS)
            if result.get("type") != "tool_calls":
                if not called_tool and self._needs_grounding():
                    called_tool = True
                    self._forced_retrieve(messages, query, sources, contexts)
                    continue
                return {"answer": result.get("text", ""),
                        "sources": _dedupe(sources), "contexts": contexts,
                        "handoff": handoff}

            calls = result.get("calls", [])
            # 提前终止: 当前轮要求的工具全部已经调用过, 且已有非空检索结果 →
            # 模型在重复检索, 直接让它基于已有上下文作答, 省一轮 LLM 调用
            if (called_tool and sources and
                    all(c.get("name") in called_tool_names for c in calls)):
                logger.info("Agentic[%s] 提前终止: 第%d轮工具(%s)均已调用过且已有上下文",
                            self.intent, turn + 1, [c.get("name") for c in calls])
                # 给模型一个"工具结果已提供, 请直接作答"的提示, 然后下一轮它会直接回答
                append_tool_result(messages, calls[0],
                                   "以上信息已在之前的工具调用中提供, 请基于已有上下文直接回答用户问题。")
                continue

            assistant_msg = {"role": "assistant", "content": None, "tool_calls": []}
            for call in calls:
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

            # 并行执行多个工具调用 (工具执行不涉及 LLM, 并行不影响正确性)
            if len(calls) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as pool:
                    futures = {pool.submit(self._execute_tool, call): i for i, call in enumerate(calls)}
                    results = [None] * len(calls)
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
            else:
                results = [self._execute_tool(calls[0])] if calls else []

            for i, call in enumerate(calls):
                out = results[i]
                called_tool = True
                called_tool_names.add(call.get("name", ""))
                append_tool_result(messages, call, out["text"])
                sources.extend(out["sources"])
                handoff = handoff or bool(out.get("handoff"))
                self._merge_contexts(contexts, out.get("contexts"))
            logger.info("Agentic[%s] 工具循环第 %d 轮: 调用 %s (并行=%s)",
                        self.intent, turn + 1,
                        [c.get("name") for c in calls], len(calls) > 1)

        logger.warning("Agentic[%s] 工具循环达到上限 %d, 返回占位", self.intent, MAX_TOOL_TURNS)
        return {"answer": "工具调用次数过多，请简化问题后重试。",
                "sources": _dedupe(sources), "contexts": contexts,
                "handoff": handoff}

    def _gather_context(self, query: str, on_event=None) -> dict:
        """仅收集工具上下文 (不生成最终答案), 供流式模式使用.
        优化: (1) 重复调用相同工具且已有上下文时提前终止; (2) 多工具并行执行.
        """
        messages = [{"role": "system", "content": self.system_prompt + "\n\n" + TOOL_GUIDE},
                    {"role": "user", "content": query}]
        ctx_texts, sources, contexts = [], [], {"local": [], "global": [], "graph": []}
        handoff = False
        called_tool = False
        called_tool_names = set()

        for turn in range(MAX_TOOL_TURNS):
            result = self.client.chat_with_tools(messages, TOOL_DEFINITIONS)
            if result.get("type") != "tool_calls":
                if not called_tool and self._needs_grounding():
                    self._forced_retrieve(messages, query, sources, contexts)
                break

            calls = result.get("calls", [])
            # 提前终止: 重复调用相同工具且已有上下文
            if (called_tool and sources and
                    all(c.get("name") in called_tool_names for c in calls)):
                logger.info("Agentic[%s] gather提前终止: 第%d轮工具均已调用过",
                            self.intent, turn + 1)
                break

            assistant_msg = {"role": "assistant", "content": None, "tool_calls": []}
            for call in calls:
                assistant_msg["tool_calls"].append({
                    "id": call.get("id", ""), "type": "function",
                    "function": {"name": call.get("name", ""),
                                 "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)},
                    "_raw_function_call": call.get("_raw_function_call"),
                })
            messages.append(assistant_msg)

            # 并行执行工具
            if len(calls) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as pool:
                    futures = {pool.submit(self._execute_tool, call): i for i, call in enumerate(calls)}
                    results = [None] * len(calls)
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
            else:
                results = [self._execute_tool(calls[0])] if calls else []

            for i, call in enumerate(calls):
                if on_event:
                    on_event({"type": "stage", "stage": "tool_call",
                              "message": f"\U0001f527 调用工具 {call.get('name')}..."})
                out = results[i]
                called_tool = True
                called_tool_names.add(call.get("name", ""))
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

    # ── 接地护栏 ─────────────────────────────────────

    # 无需知识库依据的意图: 闲聊/问候直接答, 不该被强制检索
    _NO_GROUNDING_INTENTS = {"chitchat", "greeting"}

    def _needs_grounding(self) -> bool:
        return self.intent not in self._NO_GROUNDING_INTENTS

    def _forced_retrieve(self, messages: list, query: str, sources: list,
                         contexts: dict) -> None:
        """模型跳过工具时, 确定性补一次 retrieve_knowledge, 并把原文喂回去重答。

        为什么必须有这道护栏: 模型不调工具时手里没有任何原文, 而提示词要求"标注来源",
        实测它就会照提示词的范例编一个来源名出来 —— 幻觉引用比没有引用更坏。
        用 retrieve_knowledge 而不是预取上下文: 前者带实体对齐抑制 (见 _answer_fallback)。
        """
        out = self._execute_tool({"name": "retrieve_knowledge", "id": "guard-retrieve",
                                  "arguments": {"query": query}})
        sources.extend(out.get("sources") or [])
        self._merge_contexts(contexts, out.get("contexts"))
        messages.append({"role": "user", "content": f"{out.get('text', '')}\n\n"
                         + _GROUNDING_REMINDER})
        logger.info("Agentic[%s] 模型未调工具即作答 → 已确定性补检索", self.intent)

    @staticmethod
    def _merge_contexts(contexts: dict, ctx: Optional[dict]) -> None:
        if not ctx:
            return
        contexts["local"].extend(ctx.get("local", []))
        contexts["graph"].extend(ctx.get("graph", []))
        if ctx.get("global"):
            contexts["global"] = ctx.get("global")

    # ── 回退路径 ─────────────────────────────────────

    def _answer_fallback(self, query: str, prefetched_contexts: Optional[dict]) -> dict:
        """预取检索 + 普通 chat (与旧 domain agent 行为一致)"""
        # 一律走 retrieve_knowledge: 统一实体对齐抑制 (知识库外的实体不喂其他公司内容)
        # 不能用 prefetched_contexts 直接格式化, 否则绕过抑制
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
