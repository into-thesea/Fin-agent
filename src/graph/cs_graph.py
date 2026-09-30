"""
P3: 金融理财产品智能客服 LangGraph 多 Agent 状态机

把单线管道 (Router → AgenticAgent → Reviewer → Formatter) 重构为 LangGraph StateGraph:
  classify → route_retrieve → slot_check → orchestrator → domain → reviewer → (regenerate 循环) → formatter

复用: RouterAgent / AgenticAgent / ReviewerAgent / slot_filler / format_response 全部现有组件，零重写。
新增: LangGraph 状态机、orchestrator 节点（多 Agent 派发入口）、checkpointer 会话状态、
      确定性转人工节点 (疑似诈骗/投诉, 不依赖 LLM 自觉, 直接转人工留痕)。

运行:
  graph = build_graph()
  result = graph.invoke(init_state, config={"configurable": {"thread_id": session_id}})
"""

from __future__ import annotations

import os
import sys
import re
import logging
import concurrent.futures
from typing import TypedDict, Optional

# 直接运行本文件时把项目根加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

from src.llm.query_router import classify as rule_classify
from src.llm.llm_client import create_client
from src.agents.router_agent import RouterAgent, QueryIntent
from src.agents.agentic_agent import AgenticAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.prompts import (
    FINANCE_KB_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT,
    CS_CHITCHAT_SYSTEM_PROMPT, UNKNOWN_SYSTEM_PROMPT,
    BOUNDARY_BLOCK_LIGHT,
)
from src.core.slot_filler import extract_slots, missing_required, followup_question
from src.tools.registry import retrieve_knowledge
from src.memory.memory_store import memory_store
from src.business import finance_services as fs  # 阶段二: 适当性门控 / 申购
from src.analyst_agent import format_response  # 复用现有格式化器

MAX_RETRY_ON_REJECT = 1  # 限制重写次数: 第二次重写成功率低, 省3-8秒延迟

# ── 意图分组 & 意图 → 领域节点 映射 ──────────────
KB_INTENTS = {
    QueryIntent.PRODUCT_CONSULT, QueryIntent.PRODUCT_COMPARE,
    QueryIntent.RISK_SUITABILITY, QueryIntent.INCOME_QUESTION,
    QueryIntent.DEPOSIT_INSURANCE, QueryIntent.BUY_PROCESS,
    QueryIntent.HOLD_REDEEM, QueryIntent.FEE_RULE,
}

# 确定性转人工意图: 答案固定 + 触发 handoff, 不做 LLM 幻觉审核
HANDOFF_INTENTS = {QueryIntent.FRAUD_REPORT, QueryIntent.COMPLAINT}

INTENT_NODE_MAP = {
    QueryIntent.PRODUCT_CONSULT: "finance_node",
    QueryIntent.PRODUCT_COMPARE: "finance_node",
    QueryIntent.RISK_SUITABILITY: "finance_node",
    QueryIntent.INCOME_QUESTION: "finance_node",
    QueryIntent.DEPOSIT_INSURANCE: "finance_node",
    QueryIntent.BUY_PROCESS: "finance_node",
    QueryIntent.HOLD_REDEEM: "finance_node",
    QueryIntent.FEE_RULE: "finance_node",
    QueryIntent.FRAUD_REPORT: "handoff_node",
    QueryIntent.COMPLAINT: "handoff_node",
    QueryIntent.SERVICE_POLICY: "finance_node",   # 服务规则在知识库里(转人工条件/隐私), 走 KB 依据
    QueryIntent.CHITCHAT: "chitchat_node",
}

DOMAIN_NODE_PROMPTS = {
    "finance_node": FINANCE_KB_SYSTEM_PROMPT,
    "chitchat_node": CS_CHITCHAT_SYSTEM_PROMPT,
    "general_node": GENERAL_SYSTEM_PROMPT,
}

# 确定性转人工模板 (静态话术; 双录/留痕由上层 handoff 工单机制承载)
_HANDOFF_TEMPLATES = {
    "fraud_report": (
        "您可能遇到了疑似虚假理财或诈骗情形。请先停止一切转账与付款，"
        "切勿向任何人提供短信验证码或银行卡密码；如已转账，请尽快联系开户行申请冻结/止付，"
        "并保留聊天与转账证据。该情形涉及资金安全，我已为您转接人工坐席协助核实与留痕处理。"
    ),
    "complaint": (
        "已为您登记投诉并转接人工坐席，坐席会在 1 个工作日内响应，处理全程留痕可追溯。"
        "请保持联系方式畅通，也可继续补充描述，坐席将据此加快处理。"
    ),
}

# 申购续轮补验证码: 识别"验证码 123456 / 验证码是 123456"这类 6 位码
_CODE_RE = re.compile(r"\b(\d{6})\b")


def _should_enter_subscribe(intent: QueryIntent, slots: dict, query: str = "") -> bool:
    """是否进入申购门控(执行下单), 而非产品分析?

    语义: 用户在聊天里说"想申购/想买某产品", 意图是让他更深入了解这款产品
    (类型/风险/收益口径/适不适合自己), **不是**让系统因为他一句话就去下单。
    只有两种情况才构成"下单"并进门控(门控内仍有适当性+短信验证码二次把关):
      1) 明确购买意图(BUY_PROCESS) 且 已给定金额 —— 已构成可执行的下单指令;
      2) 续轮携带 6 位验证码(命中) 且已有目标产品 —— 上一轮门控索码后的确认轮。
    其余情形(只有产品名、只想"怎么买/流程/能不能买"、咨询中顺带金额)一律落到
    finance_node 先做产品分析, 由分析结尾引导用户确认, 不擅自进入执行。
    """
    if intent in HANDOFF_INTENTS:
        return False
    slots = slots or {}
    code_hit = bool(_CODE_RE.search(query or ""))
    product = slots.get("product")
    amount_given = slots.get("amount") is not None
    order = intent == QueryIntent.BUY_PROCESS and amount_given
    confirm = intent == QueryIntent.PRODUCT_CONSULT and code_hit  # 索码后的补码轮
    return bool(product) and (order or confirm)

# SSE 节点阶段提示文案 (前端展示)
_STAGE_MSG = {
    "classify_node": "🔍 正在分析问题意图...",
    "route_retrieve_node": "📚 正在检索知识库...",
    "slot_check_node": "🧩 检查会话槽位...",
    "orchestrator_node": "🤖 多 Agent 编排中...",
    "finance_node": "💹 理财 Agent 处理中...",
    "handoff_node": "🙋 转人工坐席中...",
    "subscribe_node": "🔐 申购门控(适当性/验证码)...",
    "chitchat_node": "💬 闲聊 Agent 处理中...",
    "general_node": "🤖 通用 Agent 处理中...",
    "reviewer_node": "✅ 正在验证回答准确性...",
    "regenerate_node": "🔁 正在重试修正...",
    "formatter_node": "✍️ 正在格式化答案...",
    "greeting_node": "👋 快速问候...",
    "simple_fact_node": "📄 快速问答...",
    "unknown_node": "🤔 处理中...",
    "memory_node": "🧠 读取记忆...",
}


class CsState(TypedDict):
    query: str
    session_id: str
    user_id: str
    slots: dict
    qclass: str
    intent: str
    confidence: float
    entities: list
    contexts: dict
    memory_context: str
    agent_result: dict
    review: dict
    answer: str
    sources: list
    handoff: bool
    retry_count: int
    stage: str


def _empty_state(query: str, session_id: str = "", user_id: str = "", slots: Optional[dict] = None) -> dict:
    return {
        "query": query, "session_id": session_id, "user_id": user_id, "slots": dict(slots or {}),
        "qclass": "", "intent": "", "confidence": 0.0, "entities": [],
        "contexts": {}, "memory_context": "",
        "agent_result": {}, "review": {}, "answer": "", "sources": [], "handoff": False,
        "retry_count": 0, "stage": "start",
    }


def build_graph(token_queue=None):
    """构建 LangGraph 状态机（组件在闭包中复用，不重复实例化）

    Args:
        token_queue: 可选。传入后每个节点开始时推送 stage 事件、
                    领域节点流式推送 token（供 SSE 消费）。
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver

    router = RouterAgent()
    reviewer = ReviewerAgent()

    def _staged(node_fn, name):
        """包装节点: 开始时推送 stage 事件"""
        def wrapped(state):
            if token_queue is not None:
                token_queue.put({"type": "stage", "stage": name,
                                 "message": _STAGE_MSG.get(name, f"节点 {name}")})
            return node_fn(state)
        return wrapped

    def _with_memory(state: dict) -> str:
        """P4: 前置记忆上下文到用户问题"""
        q = state["query"]
        ctx = state.get("memory_context") or ""
        if ctx:
            q = f"{ctx}\n\n【用户问题】\n{q}"
        return q

    # ── 1. 规则路由 (毫秒级, 决定快速通道 vs 完整管道) ──
    def classify_node(state: dict) -> dict:
        qclass = rule_classify(state["query"])
        logger.info("P3 classify: %s", qclass)
        return {"qclass": qclass, "stage": "classify"}

    def _classify_next(state: dict) -> str:
        return {"greeting": "greeting_node", "simple_fact": "simple_fact_node"}.get(state["qclass"], "route_retrieve_node")

    # ── 1.5 分层记忆注入 (P4): 会话窗口 + 用户画像 + 向量语义记忆 → memory_context ──
    def memory_node(state: dict) -> dict:
        ctx = memory_store.build_memory_context(
            state.get("session_id", ""), state.get("user_id", ""), state["query"],
        )
        if ctx:
            logger.info("P4 记忆注入: %d 字符", len(ctx))
        return {"memory_context": ctx, "stage": "memory"}

    # ── 2. 快速通道 (greeting / simple_fact) ──
    def greeting_node(state: dict) -> dict:
        messages = [
            {"role": "system", "content": "你是一个友好的金融理财产品客服助手。请用中文简洁回答。\n\n" + BOUNDARY_BLOCK_LIGHT},
            {"role": "user", "content": state["query"]},
        ]
        if token_queue is not None:
            # 流式: 逐token推送并累积, 结束后返回完整答案
            _parts = []
            for _tok in create_client(cheap=True).chat_stream(messages):
                _parts.append(_tok)
                token_queue.put({"type": "token", "token": _tok})
            ans = "".join(_parts)
        else:
            ans = create_client(cheap=True).chat(messages)
        return {"answer": ans, "sources": [], "stage": "greeting",
                "intent": QueryIntent.CHITCHAT.value, "confidence": 0.9}

    def simple_fact_node(state: dict) -> dict:
        r = retrieve_knowledge(state["query"], 8)
        messages = [
            {"role": "system", "content": "你是理财客服助手。请基于知识库检索结果简要回答, 无法确定时明确说明。\n\n" + BOUNDARY_BLOCK_LIGHT + "\n\n" + r["text"]},
            {"role": "user", "content": state["query"]},
        ]
        if token_queue is not None:
            # 流式: 逐token推送并累积, 结束后返回完整答案
            _parts = []
            for _tok in create_client(cheap=True).chat_stream(messages):
                _parts.append(_tok)
                token_queue.put({"type": "token", "token": _tok})
            ans = "".join(_parts)
        else:
            ans = create_client(cheap=True).chat(messages)
        # 快速通道用LLM分类获取intent (方案A: 提升全链路意图准确率)
        # router.route()内部先做强信号规则仲裁, 强信号命中零延迟, 其余走LLM分类
        route_result = router.route(state["query"])
        return {"answer": ans, "sources": r["sources"], "contexts": r.get("contexts", {}),
                "stage": "simple_fact", "intent": route_result.intent.value,
                "confidence": route_result.confidence}

    # ── 3. 复杂路径: 路由 + 检索 (并行) ──
    def _build_dialog_context(state: dict) -> str:
        """从会话状态构建多轮上下文摘要, 供路由模型做指代消解和意图延续判断"""
        parts = []
        slots = state.get("slots") or {}
        product = slots.get("product")
        if product:
            parts.append(f"当前会话焦点产品: {product}")
        amount = slots.get("amount")
        if amount is not None:
            parts.append(f"当前会话涉及金额: {amount}万")
        risk = slots.get("risk_level")
        if risk:
            parts.append(f"当前会话涉及风险等级: {risk}")
        if parts:
            return "【多轮对话上下文】\n" + "\n".join(parts)
        return ""

    def _intent_retrieval_consistency_check(intent, confidence, contexts):
        """意图-检索一致性校验: 路由置信度低且检索结果高度集中于另一主题时调整意图.
        只在保守条件下调整: 路由置信度 < 0.7 且 top3 来源中 >=2 个属于同一主题.
        返回 (调整后的intent, 是否调整, 调整理由).
        """
        if confidence >= 0.7:
            return intent, False, ""
        local = contexts.get("local", []) if contexts else []
        if not local:
            return intent, False, ""
        theme_rules = [
            (QueryIntent.DEPOSIT_INSURANCE, ["存款保险", "deposit_insurance", "偿付", "保障范围"]),
            (QueryIntent.RISK_SUITABILITY, ["适当性", "风险等级", "risk_level", "测评"]),
            (QueryIntent.FEE_RULE, ["费用", "费率", "fee", "手续费"]),
            (QueryIntent.HOLD_REDEEM, ["赎回", "持有", "到期", "redeem", "hold"]),
            (QueryIntent.BUY_PROCESS, ["购买", "申购", "流程", "buy", "冷静期", "双录"]),
            (QueryIntent.INCOME_QUESTION, ["收益", "利息", "业绩比较基准", "income", "保本"]),
        ]
        theme_counts = {}
        for chunk in local[:3]:
            src = (chunk.get("source") or "").lower()
            text = (chunk.get("text") or "")[:200].lower()
            for theme_intent, keywords in theme_rules:
                if any(kw.lower() in src or kw.lower() in text for kw in keywords):
                    theme_counts[theme_intent] = theme_counts.get(theme_intent, 0) + 1
                    break
        if not theme_counts:
            return intent, False, ""
        dominant_theme, count = max(theme_counts.items(), key=lambda x: x[1])
        if count >= 2 and dominant_theme != intent:
            logger.info("意图-检索一致性调整: %s(%.0f%%) -> %s (检索top3中%d篇属该主题)",
                        intent.value, confidence * 100, dominant_theme.value, count)
            return dominant_theme, True, f"检索结果高度集中于{dominant_theme.value}主题"
        return intent, False, ""

    def route_retrieve_node(state: dict) -> dict:
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever()
        dialog_ctx = _build_dialog_context(state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            route_future = pool.submit(router.route, state["query"], dialog_ctx)
            retrieve_future = pool.submit(retriever.hybrid_retrieve, state["query"])
            route_result = route_future.result()
            contexts = retrieve_future.result()
        adjusted_intent, adjusted, reason = _intent_retrieval_consistency_check(
            route_result.intent, route_result.confidence, contexts)
        if adjusted:
            route_result.intent = adjusted_intent
            route_result.explanation = (route_result.explanation or "") + f" [一致性校验调整: {reason}]"
        logger.info("P3 route: %s (%.0f%%) - %s", route_result.intent.value,
                    route_result.confidence * 100, route_result.explanation)
        return {
            "intent": route_result.intent.value,
            "confidence": route_result.confidence,
            "entities": route_result.entities,
            "contexts": contexts,
            "stage": "route_retrieve",
        }

    def _route_next(state: dict) -> str:
        return "unknown_node" if state.get("intent") == QueryIntent.UNKNOWN.value else "slot_check_node"

    def unknown_node(state: dict) -> dict:
        ans = create_client(cheap=True).chat([
            {"role": "system", "content": UNKNOWN_SYSTEM_PROMPT},
            {"role": "user", "content": state["query"]},
        ])
        return {"answer": ans, "sources": [], "stage": "unknown"}

    # ── 4. 槽位抽取 + 缺失追问 (阶段二接入资金动账流程后在此补必填槽位) ──
    def slot_check_node(state: dict) -> dict:
        intent = QueryIntent(state["intent"])
        slots = dict(state["slots"])
        slots.update(extract_slots(state["query"], state.get("entities", [])))
        missing = missing_required(intent.value, slots)
        if missing:
            logger.info("P3 slot 预检: 意图=%s 缺 %s → 追问", intent.value, missing)
            return {"slots": slots, "answer": followup_question(missing),
                    "sources": [], "stage": "slot_followup", "intent": intent.value}
        return {"slots": slots, "stage": "slot_check"}

    def _slot_next(state: dict) -> str:
        return END if state.get("stage") == "slot_followup" else "orchestrator_node"

    # ── 5. Orchestrator: 按意图派发到领域节点 ──
    def orchestrator_node(state: dict) -> dict:
        return {"stage": "orchestrator"}

    def _orchestrator_next(state: dict) -> str:
        intent = QueryIntent(state.get("intent") or QueryIntent.UNKNOWN)
        if intent in HANDOFF_INTENTS:
            return "handoff_node"   # 疑似诈骗/投诉 → 确定性转人工, 不赌 LLM 自觉
        # 申购先分析: 只有"明确产品+金额(下单)"或"续轮补验证码"才进门控, 否则落到领域节点分析
        if _should_enter_subscribe(intent, state.get("slots") or {}, state.get("query", "")):
            return "subscribe_node"
        return INTENT_NODE_MAP.get(intent, "general_node")

    # ── 5.5 确定性转人工节点 (静态话术 + handoff 标记) ──
    def handoff_node(state: dict) -> dict:
        intent = QueryIntent(state.get("intent") or QueryIntent.UNKNOWN)
        msg = _HANDOFF_TEMPLATES.get(intent.value, "已为您转接人工坐席，坐席将尽快为您处理。")
        logger.info("P3 确定性转人工: intent=%s → handoff", intent.value)
        return {"agent_result": {"answer": msg, "sources": [], "contexts": {}, "handoff": True},
                "stage": "handoff_node"}

    # ── 5.6 申购确定性门控节点 (阶段二): 适当性 → 金额 → 验证码二次确认 → 写持仓+留痕
    def subscribe_node(state: dict) -> dict:
        from src.business.context import set_current_user
        set_current_user(state.get("user_id") or None)  # P8: 绑定 user_id (持仓/申购归属)
        slots = dict(state.get("slots") or {})
        slots.update(extract_slots(state["query"], state.get("entities", [])))
        product = slots.get("product") or ""
        try:
            amount = float(slots["amount"]) if slots.get("amount") is not None else None
        except (TypeError, ValueError):
            amount = None
        m = _CODE_RE.search(state["query"])
        code = m.group(1) if m else (slots.get("code") or "")
        res = fs.subscribe_gate(state.get("user_id") or "", product, amount, code)
        logger.info("P3 申购门控: product=%s amount=%s → ok=%s handoff=%s need_code=%s",
                    product, amount, res["success"], res["handoff"], res.get("need_code"))
        return {"agent_result": {"answer": res["text"], "sources": [], "contexts": {},
                                 "handoff": bool(res["handoff"])},
                "stage": "subscribe_node"}

    # ── 6. 领域节点 (复用 AgenticAgent, prefetched_contexts 避免重复检索) ──
    def _make_domain_node(node_name: str):
        prompt = DOMAIN_NODE_PROMPTS[node_name]
        intent = {"finance_node": "finance_kb", "chitchat_node": "chitchat",
                  "general_node": "general"}[node_name]

        def node(state: dict) -> dict:
            # P8: 图节点流转绑定 user_id (工具调用经拦截器读取 thread-local)
            from src.business.context import set_current_user
            set_current_user(state.get("user_id") or None)
            agent = AgenticAgent(system_prompt=prompt, intent=intent)
            q = _with_memory(state)  # P4: 前置记忆上下文
            if token_queue is not None:
                # 流式: 工具循环收集上下文 → 逐 token 推送
                result = {}
                for token_or_result, is_done in agent.answer_stream(
                    q, prefetched_contexts=state.get("contexts"),
                ):
                    if is_done:
                        result = token_or_result
                    else:
                        token_queue.put({"type": "token", "token": token_or_result})
                if not result:
                    result = agent.answer(query=q, prefetched_contexts=state.get("contexts"))
            else:
                result = agent.answer(query=q, prefetched_contexts=state.get("contexts"))
            logger.info("P3 %s 完成, handoff=%s", node_name, bool(result.get("handoff")))
            return {"agent_result": result, "stage": node_name}

        return node

    domain_nodes = {name: _make_domain_node(name) for name in DOMAIN_NODE_PROMPTS}

    # ── 7. Reviewer 反幻觉 + 重试循环 ──
    def reviewer_node(state: dict) -> dict:
        agent_result = state["agent_result"]
        contexts = agent_result.get("contexts") or state.get("contexts") or {}
        intent = QueryIntent(state.get("intent") or QueryIntent.UNKNOWN)
        quick = reviewer.quick_check(agent_result.get("answer", ""), contexts)
        # 路由置信度只代表"意图分对了没", 不代表"答案事实站得住", 不能当跳过事实审核的凭据。
        # 跳过 LLM 复核的条件 (满足任一即可):
        #   a) 本地 quick_check 通过 (数字/来源可接地)
        #   b) 确定性静态节点 (转人工/申购门控)
        #   c) 低风险意图: 闲聊/服务规则/未知兜底 — 这些回答不存在收益承诺等高危幻觉,
        #      quick_check 的禁语+PII检查已足够, 不需要再调一次 LLM (省1-2秒)
        _LOW_RISK_INTENTS = {QueryIntent.CHITCHAT, QueryIntent.SERVICE_POLICY, QueryIntent.UNKNOWN}
        skip_llm = (quick.verdict == "pass" or intent in HANDOFF_INTENTS
                    or state.get("stage") in ("subscribe_node", "handoff_node")
                    or intent in _LOW_RISK_INTENTS)
        if skip_llm:
            review = quick
            review.overall_score = 85
            logger.info("P3 跳过 LLM 审核 (QuickCheck/确定性转人工/高置信)")
        else:
            review = reviewer.review(agent_result.get("answer", ""), contexts)
        agent_result["_review"] = review
        return {
            "agent_result": agent_result,
            "review": {"verdict": review.verdict, "score": review.overall_score,
                       "claims": len(review.claims) if review.claims else 0},
            "stage": "reviewer",
        }

    def _review_next(state: dict) -> str:
        if state["review"].get("verdict") == "reject" and state.get("retry_count", 0) < MAX_RETRY_ON_REJECT:
            return "regenerate_node"
        return "formatter_node"

    def regenerate_node(state: dict) -> dict:
        """注入 Reviewer 反馈重新生成 (复用现有 _regenerate_with_feedback 逻辑)"""
        agent_result = state["agent_result"]
        contexts = agent_result.get("contexts") or state.get("contexts") or {}
        review_obj = agent_result.get("_review")
        local_text = "\n".join(
            [f"- {c.get('content', '')[:1500]} (来源: {c.get('source', '未知')})"
             for c in contexts.get('local', [])]
        )
        issues = "\n".join([
            f"- 无支撑断言: {c.statement}"
            for c in review_obj.claims if not c.supported
        ]) if review_obj and review_obj.claims else "无具体问题"
        feedback_prompt = f"""你是一个金融理财产品客服助手。请重新回答用户问题。

⚠️ **上次回答被审核发现以下问题:**
{issues}

**修正要求:**
1. 确保所有数字、收益口径、保本属性都有原文支撑
2. 如果原文中找不到某个数字，请如实说明"未在资料中找到该数据"
3. 非存款产品必须补上"理财/基金/保险非存款、投资须谨慎"提示
4. 严格引用来源

【问题】: {state['query']}

【参考原文】:
{local_text}
"""
        answer = create_client().chat(
            messages=[{"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                      {"role": "user", "content": feedback_prompt}],
            temperature=0.05,
        )
        corrected = {
            "answer": answer,
            "sources": agent_result.get("sources", []),
            "entities": agent_result.get("entities"),
            "contexts": contexts,
        }
        retry_count = state.get("retry_count", 0) + 1
        logger.warning("P3 审核拒绝, 重试 %d/%d", retry_count, MAX_RETRY_ON_REJECT)
        return {"agent_result": corrected, "retry_count": retry_count, "stage": "regenerate"}

    # ── 8. 格式化输出 ──
    def formatter_node(state: dict) -> dict:
        agent_result = state["agent_result"]
        review_obj = agent_result.get("_review")
        formatted = format_response(agent_result, review_obj)
        return {
            "answer": formatted,
            "sources": agent_result.get("sources", []),
            "handoff": bool(agent_result.get("handoff")),
            "stage": "formatter",
        }

    # ── 构图 (所有节点经 _staged 包装, 开始时推送 stage 事件) ──
    g = StateGraph(CsState)
    g.add_node("classify_node", _staged(classify_node, "classify_node"))
    g.add_node("memory_node", _staged(memory_node, "memory_node"))
    g.add_node("greeting_node", _staged(greeting_node, "greeting_node"))
    g.add_node("simple_fact_node", _staged(simple_fact_node, "simple_fact_node"))
    g.add_node("route_retrieve_node", _staged(route_retrieve_node, "route_retrieve_node"))
    g.add_node("unknown_node", _staged(unknown_node, "unknown_node"))
    g.add_node("slot_check_node", _staged(slot_check_node, "slot_check_node"))
    g.add_node("orchestrator_node", _staged(orchestrator_node, "orchestrator_node"))
    g.add_node("handoff_node", _staged(handoff_node, "handoff_node"))
    g.add_node("subscribe_node", _staged(subscribe_node, "subscribe_node"))
    for name in domain_nodes:
        g.add_node(name, _staged(domain_nodes[name], name))
    g.add_node("reviewer_node", _staged(reviewer_node, "reviewer_node"))
    g.add_node("regenerate_node", _staged(regenerate_node, "regenerate_node"))
    g.add_node("formatter_node", _staged(formatter_node, "formatter_node"))

    g.add_edge(START, "classify_node")
    g.add_edge("classify_node", "memory_node")  # P4: 路由前注入记忆
    g.add_conditional_edges("memory_node", _classify_next,
                            {"greeting_node": "greeting_node", "simple_fact_node": "simple_fact_node",
                             "route_retrieve_node": "route_retrieve_node"})
    g.add_edge("greeting_node", END)
    g.add_edge("simple_fact_node", END)
    g.add_conditional_edges("route_retrieve_node", _route_next,
                            {"unknown_node": "unknown_node", "slot_check_node": "slot_check_node"})
    g.add_edge("unknown_node", END)
    g.add_conditional_edges("slot_check_node", _slot_next,
                            {END: END, "orchestrator_node": "orchestrator_node"})
    # orchestrator 按意图派发到领域/转人工节点 (返回值为节点名 → 恒等映射)
    domain_targets = {name: name for name in set(INTENT_NODE_MAP.values()) | {"general_node", "subscribe_node"}}
    g.add_conditional_edges("orchestrator_node", _orchestrator_next, domain_targets)
    # 领域节点与确定性转人工节点统一汇入 reviewer
    for name in domain_nodes:
        g.add_edge(name, "reviewer_node")
    g.add_edge("handoff_node", "reviewer_node")
    g.add_edge("subscribe_node", "reviewer_node")
    g.add_conditional_edges("reviewer_node", _review_next,
                            {"regenerate_node": "regenerate_node", "formatter_node": "formatter_node"})
    g.add_edge("regenerate_node", "reviewer_node")
    g.add_edge("formatter_node", END)

    return g.compile(checkpointer=MemorySaver())


def run(query: str, session_id: str = "", user_id: str = "", slots: Optional[dict] = None) -> dict:
    """同步运行一次对话 (快捷入口)"""
    graph = build_graph()
    result = graph.invoke(
        _empty_state(query, session_id, user_id, slots),
        config={"configurable": {"thread_id": session_id or "default"}},
    )
    # P4: 回写分层记忆
    try:
        memory_store.save_turn(session_id, user_id, query,
                               result.get("answer", ""), result.get("entities"))
    except Exception as e:
        logger.debug("记忆回写失败 (可忽略): %s", e)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "handoff": result.get("handoff", False),
        "intent": result.get("intent", ""),
        "review": result.get("review", {}),
        "stage": result.get("stage", ""),
        "_confidence": result.get("confidence"),
    }


def run_graph_stream(token_queue, query: str, session_id: str = "", user_id: str = "",
                     slots: Optional[dict] = None) -> None:
    """后台线程: 运行图并推送 stage/token/result/done 事件到 token_queue

    事件类型 (与 chat.py 现有 SSE 循环兼容):
      {"type": "stage", "stage", "message"}  — 节点阶段
      {"type": "token", "token"}             — 领域 Agent 逐 token
      {"type": "result", "answer", "sources", "handoff", "review"}
      {"type": "error", "message"}
      {"type": "done"}
    """
    import asyncio
    graph = build_graph(token_queue)
    init = _empty_state(query, session_id, user_id, slots)
    config = {"configurable": {"thread_id": session_id or "default"}}
    try:
        result = graph.invoke(init, config=config)
        # P4: 回写分层记忆
        try:
            memory_store.save_turn(session_id, user_id, query,
                                   result.get("answer", ""), result.get("entities"))
        except Exception as e:
            logger.debug("记忆回写失败 (可忽略): %s", e)
        token_queue.put({
            "type": "result",
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "handoff": bool(result.get("handoff")),
            "intent": result.get("intent", ""),
            "review": result.get("review", {}),
            "slots": result.get("slots", {}),
            "_confidence": result.get("confidence"),
        })
    except Exception as e:
        logger.error("P3 图流式运行失败: %s", e, exc_info=True)
        token_queue.put({"type": "error", "message": f"处理出错: {str(e)[:200]}"})
    finally:
        token_queue.put({"type": "done"})


if __name__ == "__main__":
    # 自检: 覆盖 KB 问答 / 存款保险多跳 / 收益口径 / 疑似诈骗转人工 (需 LLM API)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    tests = ["你好", "业绩比较基准是什么意思？", "稳盈添利30天保本吗？", "银行理财是存款吗，存款保险赔吗？", "我测评R2能买平衡增利180天吗？", "朋友推荐的内部高收益理财靠谱吗"]
    graph = build_graph()
    for q in tests:
        res = graph.invoke(_empty_state(q, session_id=f"selftest_{q[:4]}"),
                           config={"configurable": {"thread_id": "selftest"}})
        print(f"\nQ: {q}\n  intent={res.get('intent','')} stage={res.get('stage','')} handoff={res.get('handoff')}")
        print(f"  A: {(res.get('answer','') or '')[:150]}...")
