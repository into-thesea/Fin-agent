"""
Fin-Agent 分析智能体 (管道架构 v4.0)

管道流程:
  RouterAgent (意图分类)
      │
      ▼
  Domain Agent (FactAgent / TrendAgent / CompareAgent / SummaryAgent)
      │
      ▼
  ReviewerAgent (反幻觉验证)
      │
      ▼
  ResponseFormatter (输出格式化 + 证据链组装)

失败重试:
  - Reviewer 发现严重幻觉 (verdict=reject) → 自动重生成 (最多 2 次)
  - 重试时 Reviewer 的反馈作为额外上下文注入
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator, Optional, Tuple

from dotenv import load_dotenv

from src.agents.compare_agent import CompareAgent
from src.agents.fact_agent import FactAgent
from src.agents.reviewer_agent import ReviewerAgent, ReviewResult
from src.agents.router_agent import QueryIntent, RouterAgent, RoutingResult
from src.agents.trend_agent import TrendAgent
from src.core.slot_filler import extract_slots, followup_question, missing_required
from src.llm.llm_client import create_client
from src.retrieval.retriever import HybridRetriever, format_graph_context
from src.tools.web_search import WebSearchTool

# 业务意图: 阶段一(理财域)咨询/查询统一走知识库 + Reviewer 幻觉审核;
# 资金动账/确定性业务意图在阶段二接入后再归类跳过幻觉审核。
BUSINESS_INTENTS: set = set()

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

MAX_RETRY_ON_REJECT = 2  # 被 Reviewer 拒绝后最多重试次数


# ──────────────────────────────────────────────
# 通用 Agent (回退)
# ──────────────────────────────────────────────

from src.agents.prompts import GENERAL_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT


class GeneralAgent:
    """通用 Agent (未匹配到特定意图时回退)"""
    def __init__(self) -> None:
        self.client = create_client()
        self.retriever = HybridRetriever()

    def answer(self, query: str, entity_hint: Optional[str] = None, context: Optional[dict] = None,
                web_context: Optional[str] = None, prefetched_contexts: Optional[dict] = None) -> dict:
        contexts = prefetched_contexts or self.retriever.hybrid_retrieve(query, entity_hint=entity_hint)
        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} (来源: {c.get('source', '未知')})" for c in contexts['local']]
        )
        global_text = "\n".join(
            [f"- 宏观背景: {s}" for s in contexts['global']]
        )
        graph_text = format_graph_context(contexts.get('graph', []))
        web_text = f"\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""【问题】: {query}

【参考原文】:
{local_text}

【宏观背景】:
{global_text}

【图谱关系】:
{graph_text}{web_text}"""

        answer = self.client.chat(
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )
        sources = list(set(c['source'] for c in contexts['local']))

        return {
            "answer": answer,
            "sources": sources,
            "entities": entity_hint,
            "contexts": contexts,
        }

    def answer_stream(self, query: str, entity_hint: Optional[str] = None, context: Optional[dict] = None,
                       web_context: Optional[str] = None, prefetched_contexts: Optional[dict] = None) -> Iterator[Tuple[Any, bool]]:
        """流式版本, 逐 token 产出 (token, done) 二元组"""
        contexts = prefetched_contexts or self.retriever.hybrid_retrieve(query, entity_hint=entity_hint)
        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} (来源: {c.get('source', '未知')})" for c in contexts['local']]
        )
        global_text = "\n".join(
            [f"- 宏观背景: {s}" for s in contexts['global']]
        )
        graph_text = format_graph_context(contexts.get('graph', []))
        web_text = f"\n【网络补充信息】:\n{web_context}" if web_context else ""

        user_input = f"""【问题】: {query}

【参考原文】:
{local_text}

【宏观背景】:
{global_text}

【图谱关系】:
{graph_text}{web_text}"""

        full_answer = []
        sources = list(set(c['source'] for c in contexts['local']))

        for content in self.client.chat_stream(
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        ):
            full_answer.append(content)
            yield content, False

        result = {
            "answer": "".join(full_answer),
            "sources": sources,
            "entities": entity_hint,
            "contexts": contexts,
        }
        yield result, True


# ──────────────────────────────────────────────
# 响应格式化器
# ──────────────────────────────────────────────

def format_response(agent_result: dict, review_result: Any = None) -> str:
    """
    最终输出格式化 + 证据链组装

    在回答末尾追加:
      - 信心指数
      - 溯源证据列表
      - (可选) 审核备注
    """
    answer = agent_result.get("answer", "")
    sources = agent_result.get("sources", [])

    # 构建溯源脚注
    source_section = "\n\n---\n**📚 溯源证据链**\n"
    for i, s in enumerate(sources[:5], 1):
        source_section += f"\n{i}. 📄 `{s}`"

    if review_result:
        score = review_result.overall_score
        if score >= 80:
            badge = "🟢 高可信度"
        elif score >= 50:
            badge = "🟡 中等可信度"
        else:
            badge = "🔴 低可信度 — 建议核实"

        source_section += f"\n\n**🎯 信心指数**: {badge} ({score:.0f}/100)"

        if review_result.suggestion:
            source_section += f"\n**💡 审查备注**: {review_result.suggestion}"

    return answer + source_section


# ──────────────────────────────────────────────
# 管道式 AnalystAgent
# ──────────────────────────────────────────────

class AnalystAgent:
    """
    管道式分析智能体

    流程:
      Router → Domain Agent → Reviewer → Formatter
      ↓ (如果 Review=reject, 循环重试)
    """

    def __init__(self) -> None:
        self.router = RouterAgent()
        self.fact_agent = FactAgent()
        self.trend_agent = TrendAgent()
        self.compare_agent = CompareAgent()
        self.general_agent = GeneralAgent()
        self.reviewer = ReviewerAgent()
        self.retriever = HybridRetriever()
        self.web_search = WebSearchTool()

    # ──────────────────────────────────────────────
    # Web 搜索补充
    # ──────────────────────────────────────────────

    def _needs_web_supplement(self, agent_result: dict, route: RoutingResult) -> bool:
        """客服场景默认不做 Web 补充 (合规 + 稳定性)。Phase 2 需要时可基于配置开启。"""
        return False

    def _web_search_supplement(self, query: str) -> Optional[str]:
        """
        执行 Web 搜索并返回格式化后的上下文

        Returns:
            格式化的搜索结果字符串，搜索失败时返回 None
        """
        if not self.web_search.available:
            return None

        try:
            response = self.web_search.search_finance(query, max_results=5)
            if response.results:
                formatted = self.web_search.format_for_prompt(response)
                logger.info("Web 搜索返回 %d 条结果", len(response.results))
                return formatted
        except Exception as e:
            logger.warning("Web 搜索补充失败: %s", e)

        return None

    def analyze(self, query: str, state=None) -> dict:
        """
        管道式分析入口

        Args:
            query: 用户查询
            state: 可选的会话状态 (用于业务意图槽位预检)

        Returns:
            {"answer": str, "entities": str, "sources": list, "review": dict}
        """
        # 缓存命中直接返回
        from src.core.answer_cache import get_cached, set_cache
        cached = get_cached(query)
        if cached:
            return cached

        # 规则路由: 仅问候走快速通道 (零检索, <1s)
        from src.llm.llm_client import create_client
        from src.llm.query_router import classify as classify_query
        qclass = classify_query(query)

        if qclass == "greeting":
            cheap_client = create_client(cheap=True)
            answer = cheap_client.chat([
                {"role": "system", "content": "你是一个友好的客服助手。请用中文简洁回答。"},
                {"role": "user", "content": query},
            ])
            result = {
                "answer": answer,
                "entities": None,
                "entities_raw": [],
                "_intent": "greeting",
                "_confidence": 1.0,
                "_rule_result": "greeting",
                "sources": [],
                "review": {"score": 85, "verdict": "pass", "claims": 0},
            }
            try:
                set_cache(query, result)
            except Exception as e:
                logger.debug("快速通道缓存写入失败 (可忽略): %s", e)
            logger.info("快速通道 (greeting): %s", query[:40])
            return result

        logger.info("=" * 50)
        logger.info("AnalystAgent 管道启动: %s", query[:80])

        # ─── Step 1: 路由 + 检索并行 ─────────────────────
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            route_future = pool.submit(self.router.route, query)
            retrieve_future = pool.submit(self.retriever.hybrid_retrieve, query)
            route_result = route_future.result()
            prefetched_contexts = retrieve_future.result()

        # 提取实体 (客服场景不喂图谱, get_entity_hint 返回 None)
        entity_hint = self.router.get_entity_hint(route_result)

        logger.info(
            "路由: %s (置信度: %.0f%%)",
            route_result.intent.value,
            route_result.confidence * 100,
        )

        # ─── Step 1.5: UNKNOWN 意图 → 不检索, 礼貌回答 ──
        if route_result.intent == QueryIntent.UNKNOWN:
            from src.agents.prompts import UNKNOWN_SYSTEM_PROMPT
            logger.info("UNKNOWN 意图 (%.0f%%), 走礼貌回答 (不检索知识库)",
                        route_result.confidence * 100)
            answer = create_client(cheap=True).chat([
                {"role": "system", "content": UNKNOWN_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ])
            result = {
                "answer": answer,
                "entities": entity_hint,
                "entities_raw": route_result.entities,
                "_intent": route_result.intent.value,
                "_confidence": route_result.confidence,
                "_rule_result": qclass,
                "sources": [],
                "review": {"score": 85, "verdict": "pass", "claims": 0},
            }
            try:
                set_cache(query, result)
            except Exception as e:
                logger.debug("UNKNOWN 回答缓存写入失败 (可忽略): %s", e)
            return result

        # ─── Step 1.7: 业务意图槽位预检 (缺必需槽位 → 追问, 不进入 Agent) ──
        if route_result.intent in BUSINESS_INTENTS and state is not None:
            state.slots.update(extract_slots(query, route_result.entities))
            missing = missing_required(route_result.intent.value, state.slots)
            if missing:
                state.awaiting_slot = missing[0]
                logger.info("槽位预检: 意图=%s 缺 %s → 追问", route_result.intent.value, missing)
                return {
                    "answer": followup_question(missing),
                    "entities": entity_hint,
                    "entities_raw": route_result.entities,
                    "_intent": route_result.intent.value,
                    "_confidence": route_result.confidence,
                    "_rule_result": qclass,
                    "sources": [],
                    "review": {"score": 85, "verdict": "pass", "claims": 0},
                    "handoff": False,
                    "_followup": True,
                }

        # ─── Step 2: 领域 Agent (传入预取上下文, 避免重复检索) ──
        agent_result = self._route_to_domain(query, route_result, entity_hint,
                                              prefetched_contexts=prefetched_contexts)

        # ─── Step 2.5: Web 搜索补充 (本地知识不足时) ────
        web_context = None
        if self._needs_web_supplement(agent_result, route_result):
            logger.info("本地知识不足，启动 Web 搜索补充...")
            web_context = self._web_search_supplement(query)

        # ─── Step 3: 审核 ───────────────────────────────
        # 业务意图 (订单/物流/退款/投诉): 答案来自确定性业务工具, 跳过幻觉审核
        if route_result.intent in BUSINESS_INTENTS:
            review = ReviewResult(overall_score=85, verdict="pass",
                                  suggestion="业务工具结果, 已跳过幻觉审核")
            agent_result["_review"] = review
            final_result = agent_result
        # 高置信度路由 (>=85%) + 本地数据充足 → 跳过 LLM 审核
        elif route_result.confidence >= 0.85 and not web_context:
            quick = self.reviewer.quick_check(agent_result["answer"], agent_result.get("contexts", {}))
            if quick.verdict == "pass":
                logger.info("高置信度路由 + QuickCheck 通过, 跳过 LLM 审核")
                review = quick
                review.overall_score = 85
                agent_result["_review"] = review
                final_result = agent_result
            else:
                final_result = self._review_and_retry(query, agent_result, web_context=web_context)
        else:
            final_result = self._review_and_retry(query, agent_result, web_context=web_context)

        # ─── Step 4: 格式化 ──────────────────────────────
        formatted = format_response(
            final_result,
            final_result.get("_review"),
        )

        # 构造返回 (兼容旧接口)
        result = {
            "answer": formatted,
            "entities": entity_hint,
            "entities_raw": route_result.entities,  # DST 用: 实体列表
            "_intent": route_result.intent.value,   # DST 用: 意图
            "_confidence": route_result.confidence, # 审计用: 路由置信度
            "_rule_result": qclass,                  # 审计用: 规则路由分类
            "sources": final_result.get("sources", []),
            "review": {
                "score": final_result.get("_review", {}).overall_score if final_result.get("_review") else None,
                "verdict": final_result.get("_review", {}).verdict if final_result.get("_review") else None,
                "claims": len(final_result.get("_review", {}).claims) if final_result.get("_review") else 0,
            },
            "handoff": bool(final_result.get("handoff")),
        }
        # 写入缓存
        try:
            set_cache(query, result)
        except Exception as e:
            logger.debug("分析结果缓存写入失败 (可忽略): %s", e)
        return result

    def analyze_stream(self, token_queue: Any, query: str, state=None) -> None:
        """
        流式分析入口 — 在后台线程中运行, 通过 token_queue 推送 (type, data)

        token_queue.put({"type": "stage", "stage": "...", "message": "..."})
        token_queue.put({"type": "token", "token": "..."})
        token_queue.put({"type": "result", ...})
        token_queue.put({"type": "done"})
        """
        try:
            # Step 1: 路由 + 检索并行
            token_queue.put({"type": "stage", "stage": "routing",
                             "message": "🔍 正在分析问题意图..."})
            token_queue.put({"type": "stage", "stage": "retrieving",
                             "message": "📚 正在检索金融知识库..."})

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                route_future = pool.submit(self.router.route, query)
                retrieve_future = pool.submit(self.retriever.hybrid_retrieve, query)
                route_result = route_future.result()
                prelim = retrieve_future.result()

            entity_hint = self.router.get_entity_hint(route_result)

            # Step 2: 领域 Agent (流式)
            token_queue.put({"type": "stage", "stage": "analyzing",
                             "message": "🧠 正在生成分析结果..."})

            from src.agents.agentic_agent import AgenticAgent
            from src.agents.prompts import (
                CS_CHITCHAT_SYSTEM_PROMPT,
                FINANCE_KB_SYSTEM_PROMPT,
                GENERAL_SYSTEM_PROMPT,
                UNKNOWN_SYSTEM_PROMPT,
            )

            # UNKNOWN 意图 → 礼貌回答, 不检索知识库
            if route_result.intent == QueryIntent.UNKNOWN:
                from src.llm.llm_client import create_client
                logger.info("UNKNOWN 意图 (%.0f%%), 走礼貌回答 (不检索)",
                            route_result.confidence * 100)
                full = []
                for chunk in create_client(cheap=True).chat_stream([
                    {"role": "system", "content": UNKNOWN_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]):
                    token_queue.put({"type": "token", "token": chunk})
                    full.append(chunk)
                token_queue.put({"type": "result", "answer": "".join(full),
                                 "sources": [], "entities": entity_hint,
                                 "review": {"score": 85, "verdict": "pass", "claims": 0}})
                return

            # 业务意图槽位预检 (缺必需槽位 → 追问)
            if route_result.intent in BUSINESS_INTENTS and state is not None:
                state.slots.update(extract_slots(query, route_result.entities))
                missing = missing_required(route_result.intent.value, state.slots)
                if missing:
                    state.awaiting_slot = missing[0]
                    token_queue.put({"type": "result",
                                     "answer": followup_question(missing),
                                     "sources": [], "entities": entity_hint,
                                     "review": {"score": 85, "verdict": "pass", "claims": 0},
                                     "handoff": False})
                    return

            prompt_map = {
                QueryIntent.PRODUCT_CONSULT: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.PRODUCT_COMPARE: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.RISK_SUITABILITY: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.INCOME_QUESTION: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.DEPOSIT_INSURANCE: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.BUY_PROCESS: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.HOLD_REDEEM: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.FEE_RULE: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.FRAUD_REPORT: GENERAL_SYSTEM_PROMPT,
                QueryIntent.COMPLAINT: GENERAL_SYSTEM_PROMPT,
                QueryIntent.SERVICE_POLICY: FINANCE_KB_SYSTEM_PROMPT,
                QueryIntent.CHITCHAT: CS_CHITCHAT_SYSTEM_PROMPT,
            }
            domain_agent = AgenticAgent(
                system_prompt=prompt_map.get(route_result.intent, GENERAL_SYSTEM_PROMPT),
                intent=route_result.intent.value,
            )

            # 客服场景不做 Web 补充 (Phase 2 可配置开启)
            web_context = None
            needs_web = False

            # 流式生成: AgenticAgent 工具收集上下文 → 流式最终答案
            full_answer = []
            token_queue.put({"type": "result_meta",
                             "sources": list(set(c["source"] for c in prelim.get("local", []) if "source" in c)),
                             "entities": entity_hint})

            def _on_event(msg):
                if msg.get("type") == "stage":
                    token_queue.put({k: v for k, v in msg.items()})

            for token_or_result, is_done in domain_agent.answer_stream(
                query, prefetched_contexts=prelim, on_event=_on_event,
            ):
                if is_done:
                    agent_result = token_or_result
                else:
                    token_queue.put({"type": "token", "token": token_or_result})
                    full_answer.append(token_or_result)

            # Step 3: Reviewer 审核 (高置信度/QuickCheck/Web 都跳过 LLM)
            token_queue.put({"type": "stage", "stage": "reviewing",
                             "message": "✅ 正在验证回答准确性..."})
            contexts = agent_result.get("contexts", {})
            quick = self.reviewer.quick_check(agent_result["answer"], contexts)
            skip_llm = (
                quick.verdict == "pass" or web_context
                or route_result.intent in BUSINESS_INTENTS
                or route_result.confidence >= 0.85
            )
            if skip_llm:
                review = quick
                review.overall_score = 85
                logger.info("跳过 LLM 审核 (QuickCheck/Web/简单查询)")
            else:
                review = self.reviewer.review(agent_result["answer"], contexts, web_context=web_context)
            agent_result["_review"] = review

            # 格式化
            formatted = format_response(agent_result, review)
            source_list = agent_result.get("sources", [])

            token_queue.put({
                "type": "result",
                "answer": formatted,
                "sources": source_list,
                "entities": entity_hint,
                "review": {
                    "score": review.overall_score if review else None,
                    "verdict": review.verdict if review else None,
                    "claims": len(review.claims) if review else 0,
                },
                "handoff": bool(agent_result.get("handoff")),
            })

        except Exception as e:
            logger.error("流式分析失败: %s", e, exc_info=True)
            token_queue.put({"type": "error", "message": f"处理出错: {str(e)[:200]}"})
        finally:
            token_queue.put({"type": "done"})

    def _route_to_domain(self, query: str, route: RoutingResult, entity_hint: Optional[str],
                          web_context: Optional[str] = None, prefetched_contexts: Optional[dict] = None) -> dict:
        """根据意图路由到对应的 AgenticAgent (function calling 自主决策)"""
        from src.agents.agentic_agent import AgenticAgent
        from src.agents.prompts import (
            CS_CHITCHAT_SYSTEM_PROMPT,
            FINANCE_KB_SYSTEM_PROMPT,
            GENERAL_SYSTEM_PROMPT,
        )

        prompt_map = {
            QueryIntent.PRODUCT_CONSULT: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.PRODUCT_COMPARE: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.RISK_SUITABILITY: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.INCOME_QUESTION: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.DEPOSIT_INSURANCE: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.BUY_PROCESS: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.HOLD_REDEEM: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.FEE_RULE: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.FRAUD_REPORT: GENERAL_SYSTEM_PROMPT,
            QueryIntent.COMPLAINT: GENERAL_SYSTEM_PROMPT,
            QueryIntent.SERVICE_POLICY: FINANCE_KB_SYSTEM_PROMPT,
            QueryIntent.CHITCHAT: CS_CHITCHAT_SYSTEM_PROMPT,
        }
        system_prompt = prompt_map.get(route.intent, GENERAL_SYSTEM_PROMPT)
        agent = AgenticAgent(system_prompt=system_prompt, intent=route.intent.value)
        logger.info("领域 Agent: AgenticAgent(%s)", route.intent.value)

        return agent.answer(query=query, prefetched_contexts=prefetched_contexts)

    def _review_and_retry(self, query: str, agent_result: dict,
                           web_context: Optional[str] = None) -> dict:
        """
        审核回答，如果发现严重幻觉则重试

        重试策略:
          - 将 Reviewer 的反馈注入到重试 Prompt 中
          - 最多重试 MAX_RETRY_ON_REJECT 次
          - 超过次数则接受当前结果 (标注低置信度)
        """
        contexts = agent_result.get("contexts")
        if not contexts:
            # 从 retriever 获取 (兼容 domain agent 未返回 contexts 的情况)
            contexts = self.retriever.hybrid_retrieve(
                query,
                entity_hint=agent_result.get("entities"),
            )

        # 先执行 QuickCheck 预检 (正则匹配, 无需 LLM)
        quick = self.reviewer.quick_check(agent_result["answer"], contexts)
        if quick.verdict == "pass" or web_context:
            # 所有数字断言都有上下文支撑, 或回答来自 Web 搜索 (本地 KB 无法验证)
            reason = "QuickCheck 通过, " if quick.verdict == "pass" else "回答来自 Web 搜索, "
            logger.info(f"{reason}跳过 LLM 审查")
            review = quick
            review.overall_score = 85
            agent_result["_review"] = review
            return agent_result

        # 执行完整 LLM 审查 (仅本地 KB 数据)
        review = self.reviewer.review(agent_result["answer"], contexts, web_context=web_context)
        agent_result["_review"] = review

        # 如果审核通过或是未知意图，直接返回
        if review.verdict in ("pass", "flag") or MAX_RETRY_ON_REJECT == 0:
            return agent_result

        # 严重幻觉 — 重试
        for attempt in range(1, MAX_RETRY_ON_REJECT + 1):
            logger.warning(
                "审核拒绝 (attempt %d/%d): %s",
                attempt, MAX_RETRY_ON_REJECT, review.suggestion,
            )

            # 重生成 (注入 Reviewer 反馈 + Web 上下文)
            corrected = self._regenerate_with_feedback(
                query, agent_result, review, contexts, web_context=web_context
            )

            # 再次审核
            review = self.reviewer.review(corrected["answer"], contexts, web_context=web_context)
            corrected["_review"] = review

            if review.verdict != "reject":
                logger.info("重试成功 (attempt %d)", attempt)
                return corrected

        # 超过重试次数，接受最后一次结果但标记低置信度
        logger.warning("重试耗尽，接受当前结果 (低置信度)")
        review.overall_score = min(review.overall_score, 40)
        agent_result["_review"] = review
        return agent_result

    def _regenerate_with_feedback(
        self,
        query: str,
        prev_result: dict,
        review_result: Any,
        contexts: dict,
        web_context: Optional[str] = None,
    ) -> dict:
        """带 Reviewer 反馈的重新生成"""
        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} (来源: {c.get('source', '未知')})"
             for c in contexts.get('local', [])]
        )
        graph_text = "\n".join(contexts.get('graph', []))
        web_text = f"\n【网络补充信息】:\n{web_context}" if web_context else ""

        # 提取 Reviewer 指出的问题
        issues = "\n".join([
            f"- 无支撑断言: {c.statement}"
            for c in review_result.claims if not c.supported
        ]) if review_result.claims else "无具体问题"

        feedback_prompt = f"""你是一个金融分析师。请重新回答用户问题。

⚠️ **上次回答被审核发现以下问题:**
{issues}

**修正要求:**
1. 确保所有数字都有原文支撑
2. 如果原文中找不到某个数字，请如实说明"未在资料中找到该数据"
3. 严格引用来源

【问题】: {query}

【参考原文】:
{local_text}

【图谱关系】:
{graph_text}{web_text}
"""

        client = create_client()

        answer = client.chat(
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": feedback_prompt},
            ],
            temperature=0.05,
        )

        return {
            "answer": answer,
            "sources": prev_result.get("sources", []),
            "entities": prev_result.get("entities"),
            "contexts": contexts,
        }

    def close(self) -> None:
        """释放资源"""
        self.retriever.close()


if __name__ == "__main__":
    # 测试管道
    agent = AnalystAgent()
    tests = [
        "存款保险保多少钱？",
        "代销产品和自营理财有什么区别？",
        "哪些情况应该转人工坐席？",
    ]
    for q in tests:
        print(f"\n\n{'='*60}")
        print(f"Q: {q}")
        result = agent.analyze(q)
        print(f"A: {result['answer'][:200]}...")
        print(f"Review: {result['review']}")
    agent.close()
