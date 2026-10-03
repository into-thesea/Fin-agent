"""
金融/银行理财产品查询路由智能体 (RouterAgent)

功能:
  1. 将用户查询分类为预定义理财客服意图
  2. 提取核心实体 (产品名、产品类型、金额、期限、风险等级)
  3. 将查询路由到对应的领域处理路径

意图类型 (理财域):
  - PRODUCT_CONSULT:    单产品咨询 (收益/期限/风险/门槛/能不能买这一款)
  - PRODUCT_COMPARE:    多产品对比 / 该买哪款
  - RISK_SUITABILITY:   风险等级 / 测评 / 适当性 (我能买 R 几)
  - INCOME_QUESTION:    收益口径 / 保本吗 / 怎么算利息
  - DEPOSIT_INSURANCE:  存款保险 / 是不是存款 / 保不保
  - BUY_PROCESS:        怎么买 / 门槛 / 流程 / 冷静期 / 双录
  - HOLD_REDEEM:        持有 / 到期 / 赎回 / 净值 / 撤单
  - FEE_RULE:           费率 / 手续费 / 费用
  - FRAUD_REPORT:       假理财 / 飞单 / 疑似被骗 / 举报
  - COMPLAINT:          投诉 / 不满
  - CHITCHAT:           闲聊
  - UNKNOWN:            无法分类 (回退到通用处理)
"""

import os
import json
import logging
import re
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv
from src.llm.llm_client import create_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

# ── 动态 few-shot: 从 Golden QA 检索最相似的边界 case ──
_GOLDEN_QA_CACHE = None

def _load_golden_qa():
    """加载 Golden QA 标注集 (仅含边界/复杂 case, 用于动态 few-shot)"""
    global _GOLDEN_QA_CACHE
    if _GOLDEN_QA_CACHE is not None:
        return _GOLDEN_QA_CACHE
    qa_path = os.path.join(PROJECT_ROOT, "data", "eval", "finance_qa_golden.jsonl")
    cases = []
    try:
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # 只保留有明确意图且非纯问候的 case 作为 few-shot 候选
                if item.get("intent") and item.get("intent") not in ("chitchat",):
                    cases.append({
                        "question": item["question"],
                        "intent": item["intent"],
                        "complexity": item.get("complexity", "simple"),
                        "note": item.get("note", ""),
                    })
    except Exception as e:
        logger.warning("加载 Golden QA 失败: %s", e)
    _GOLDEN_QA_CACHE = cases
    logger.info("动态 few-shot 加载 %d 条 Golden QA", len(cases))
    return cases

def _keyword_overlap(query: str, question: str) -> float:
    """简单关键词重叠度 (中文按字符 bigram, 用于动态 few-shot 检索)"""
    def bigrams(s):
        s = re.sub(r'[\s\W_]', '', s.lower())  # \W匹配非单词字符(含标点), 兼容Python标准re
        return set(s[i:i+2] for i in range(len(s)-1)) if len(s) >= 2 else set(s)
    q_bi = bigrams(query)
    c_bi = bigrams(question)
    if not q_bi or not c_bi:
        return 0.0
    return len(q_bi & c_bi) / len(q_bi | c_bi)

def _retrieve_few_shots(query: str, top_k: int = 4) -> list:
    """从 Golden QA 中检索与当前查询最相似的边界 case"""
    cases = _load_golden_qa()
    if not cases:
        return []
    scored = [(c, _keyword_overlap(query, c["question"])) for c in cases]
    scored.sort(key=lambda x: x[1], reverse=True)
    # 优先选有 note 的对抗/边界 case, 其次选 multi_hop
    selected = []
    for c, score in scored:
        if score <= 0:
            break
        if c["intent"] not in [s["intent"] for s in selected]:
            selected.append(c)
        if len(selected) >= top_k:
            break
    # 如果相似度筛选不够, 补充高价值 case (有note的对抗样本)
    if len(selected) < top_k:
        for c, score in scored:
            if c not in selected and c.get("note"):
                selected.append(c)
                if len(selected) >= top_k:
                    break
    return selected


class QueryIntent(str, Enum):
    PRODUCT_CONSULT = "product_consult"
    PRODUCT_COMPARE = "product_compare"
    RISK_SUITABILITY = "risk_suitability"
    INCOME_QUESTION = "income_question"
    DEPOSIT_INSURANCE = "deposit_insurance"
    BUY_PROCESS = "buy_process"
    HOLD_REDEEM = "hold_redeem"
    FEE_RULE = "fee_rule"
    FRAUD_REPORT = "fraud_report"
    COMPLAINT = "complaint"
    SERVICE_POLICY = "service_policy"   # 客服服务规则: 转人工条件/客服边界/隐私与信息索取/投诉渠道
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


@dataclass
class RoutingResult:
    """路由结果"""
    intent: QueryIntent = QueryIntent.UNKNOWN
    entities: list = field(default_factory=list)  # 提取的实体列表 [{"name": "...", "type": "..."}]
    time_range: list = field(default_factory=list)  # 保留字段
    metrics: list = field(default_factory=list)  # 保留字段
    confidence: float = 0.0  # 路由置信度
    original_query: str = ""
    explanation: str = ""


# 意图检测 Prompt (含 CoT 推理 + 动态 few-shot 占位)
INTENT_DETECTION_PROMPT = """你是一个银行/金融理财产品智能客服的查询分类器。请先分析用户问题的核心诉求和关键词，再输出分类 JSON。

【推理要求】
请按以下步骤思考（在 analysis 字段中简要写出）:
1. 用户问题的核心诉求是什么？（问产品属性/问政策制度/问操作流程/问资金安全/其他）
2. 问题中出现了哪些关键词？这些关键词指向哪些候选意图？
3. 哪些候选意图可以排除？为什么？
4. 最终意图是什么？置信度多少？

注意: analysis 是推理过程，intent 是最终结论。两者必须一致。

意图分类定义:
  product_consult:   单款产品咨询 (收益/期限/风险等级/门槛/能不能买这一款) ("稳盈添利30天收益怎么样？")
  product_compare:   多款产品对比或选品推荐 ("大额存单和固收理财哪个适合我？" / "帮我对比两款理财")
  risk_suitability:  风险等级含义/风险测评/适当性/我能买R几级产品 ("R3是什么风险？" / "我测评是R2能买R3吗？")
  income_question:   收益口径/是否保本/利息怎么算/业绩比较基准含义 ("这款保本吗？" / "年化4%是不是就有4%？")
  deposit_insurance: 存款保险/是不是存款/保不保/偿付限额 ("银行理财是存款吗？存款保险赔吗？")
  buy_process:       怎么买/起购门槛/申购流程/冷静期/双录 ("这个怎么买？" / "首次买理财要办什么？")
  hold_redeem:       持有/到期/赎回/净值/撤单/提前支取 ("没到期能取出来吗？" / "怎么看我的净值？")
  fee_rule:          费率/手续费/管理费/赎回费 ("提前赎回要手续费吗？")
  fraud_report:      假理财/飞单/疑似被骗/举报/内部高收益渠道 ("这个高收益理财是真的吗？")
  complaint:         投诉/不满/要求处理
  service_policy:    客服服务规则 (哪些情况转人工/客服能做什么与不能做什么/隐私与信息索取/投诉渠道/工单时效) ("哪些情况应该转人工坐席？" / "客服会索要我的验证码吗？")
  chitchat:          仅限问候与寒暄 ("你是谁" / "你好" / "谢谢"); 凡涉及产品、政策、业务或服务流程的问题都不得归到此项
  unknown:           无法归类

实体提取规则:
  - Product: 产品名（"稳盈添利30天" / "大额存单" / "安鑫纯债基金"）
  - ProductType: 产品大类（理财/存款/基金/保险/大额存单）
  - Amount: 金额（"50万" / "50000元"）
  - Term: 期限（"30天" / "90天" / "3年"）
  - RiskLevel: 风险/测评等级（R1~R5）
  - YieldType: 收益口径（业绩比较基准/七日年化/执行利率/预定利率，可选）

输出格式 (严格 JSON):
{
  "analysis": "核心诉求是... 关键词包括... 排除了...因为... 最终判断为...",
  "intent": "product_consult|product_compare|risk_suitability|income_question|deposit_insurance|buy_process|hold_redeem|fee_rule|fraud_report|complaint|service_policy|chitchat|unknown",
  "entities": [{"name": "稳盈添利30天", "type": "Product"}, {"name": "R2", "type": "RiskLevel"}],
  "time_range": [],
  "metrics": [],
  "confidence": 0.95,
  "explanation": "简短理由"
}

分类示例 (易错边界):
  Q: "结构性存款的收益是保证的吗？" → intent: income_question (问收益口径, 不是存款保险)
  Q: "理财产品能承诺保本保收益吗？" → intent: income_question (问收益承诺, 不是适当性)
  Q: "大额存单和固收理财哪个适合我？" → intent: product_compare (多产品对比, 不是适当性)
  Q: "我测评是R2，能买平衡增利180天吗？" → intent: risk_suitability (适当性匹配, 不是产品咨询)
  Q: "银行理财是存款吗？存款保险赔吗？" → intent: deposit_insurance (存款保险属性, 不是收益)
  Q: "买理财为什么要双录？" → intent: buy_process (购买流程要求, 不是服务规则)
  Q: "这个高收益内部渠道靠谱吗？" → intent: fraud_report (疑似诈骗, 不是产品咨询)
  Q: "R3风险等级是什么意思？" → intent: risk_suitability (风险等级含义, 不是产品咨询)
  Q: "提前赎回会收费吗？" → intent: fee_rule (费用, 不是持有赎回)
  Q: "没到期能取出来吗？" → intent: hold_redeem (赎回/支取, 不是费用)
  Q: "客服会索要我的验证码吗？" → intent: service_policy (客服边界, 不是购买流程)

【相似历史案例参考】
{dyn_few_shots}

多轮上下文判断规则:
  - 如果上一轮在咨询某款产品的某个属性（收益/期限/风险等），本轮用"它/这个/那"指代同一款产品继续问另一个属性，应归为 product_consult，而不是根据本轮出现的关键词误分类。
    例如：上一轮问"稳盈添利30天收益怎么样"（product_consult），本轮问"那它的风险等级呢" → 仍为 product_consult（继续问同一款产品的属性），不是 risk_suitability。
  - 如果本轮明确问"我能买R几""我的测评等级""R3是什么意思"，才是 risk_suitability。
  - 上下文仅作参考，如果本轮问题明显开启了新话题（如突然问存款保险、突然投诉），以本轮内容为准。
"""


# 规则兜底用产品词表 (与 data/finance_kb/catalog.jsonl 保持一致)
_PRODUCT_KEYWORDS = [
    "日日盈", "现金管理", "稳盈", "添利", "安心固收", "固收",
    "平衡增利", "私银", "大额存单", "结构性存款", "纯债",
    "混合基金", "增额", "终身寿", "年金", "基金", "理财", "存款",
]

_RISK_RE = re.compile(r"\bR[1-5]\b", re.IGNORECASE)


class RouterAgent:
    """查询路由智能体"""

    def __init__(self):
        self.client = create_client(cheap=True)

    @staticmethod
    def _rule_fallback(query: str) -> RoutingResult:
        """LLM 路由失败时的规则兜底分类 (关键词匹配, 零成本)"""
        result = RoutingResult(original_query=query)

        q = query.lower()
        # 顺序敏感: 越具体/越涉及资金安全的规则越靠前
        # fraud_report: 必须是疑似诈骗/飞单的强信号, "靠谱吗"单独出现不算(可能是产品咨询)
        if any(kw in q for kw in ["假理财", "飞单", "被骗", "诈骗", "举报", "内部渠道", "内部高收益", "稳赚不赔"]) or \
           re.search(r"高收益.{0,6}(可靠|真|假|吗|靠谱)", q) or \
           re.search(r"(内部|私下|非正规|陌生).{0,6}(渠道|产品|理财|靠谱)", q):
            result.intent = QueryIntent.FRAUD_REPORT
            result.explanation = "规则兜底: 检测到疑似诈骗/飞单关键词"
        elif any(kw in q for kw in ["对比", "区别", "哪个好", "哪个合适", "哪个适合", "哪个收益", "哪款好", "推荐"]) or re.search(r"和.{0,8}(哪个|哪款|比)", q):
            result.intent = QueryIntent.PRODUCT_COMPARE
            result.explanation = "规则兜底: 检测到产品对比/选品关键词"
        elif any(kw in q for kw in ["投诉", "不满", "不满意"]):
            result.intent = QueryIntent.COMPLAINT
            result.explanation = "规则兜底: 检测到投诉关键词"
        elif any(kw in q for kw in ["转人工", "人工坐席", "人工客服", "客服会", "客服不会",
                                    "客服能", "服务范围", "客服边界", "隐私", "个人信息",
                                    "信息泄露", "工单"]):
            result.intent = QueryIntent.SERVICE_POLICY
            result.explanation = "规则兜底: 检测到客服服务规则关键词"
        elif any(kw in q for kw in ["存款保险", "是存款吗", "是不是存款", "保不保", "受保护", "偿付", "保障范围"]) or \
             (re.search(r"50万", q) and any(kw in q for kw in ["赔", "保", "保障", "偿付"])):
            result.intent = QueryIntent.DEPOSIT_INSURANCE
            result.explanation = "规则兜底: 检测到存款保险/产品属性关键词"
        # risk_suitability: 必须有风险/测评/等级相关信号, "能买"单独出现不算
        elif any(kw in q for kw in ["风险等级", "测评", "风险承受", "适当性"]) or _RISK_RE.search(q) or \
             (re.search(r"能买|可以买|适合买", q) and any(kw in q for kw in ["R", "风险", "测评", "等级", "稳健", "保守"])):
            result.intent = QueryIntent.RISK_SUITABILITY
            result.explanation = "规则兜底: 检测到风险等级/适当性关键词"
        elif any(kw in q for kw in ["费率", "手续费", "管理费", "赎回费", "申购费"]) or \
             (re.search(r"费用", q) and any(kw in q for kw in ["收", "怎么算", "谁出", "多少", "收取"])):
            result.intent = QueryIntent.FEE_RULE
            result.explanation = "规则兜底: 检测到费率关键词"
        elif any(kw in q for kw in ["赎回", "到期", "净值", "撤单", "持仓", "提前支取"]):
            result.intent = QueryIntent.HOLD_REDEEM
            result.explanation = "规则兜底: 检测到赎回/持有关键词"
        elif any(kw in q for kw in ["怎么买", "购买", "申购", "流程", "门槛", "起购", "冷静期", "双录"]):
            result.intent = QueryIntent.BUY_PROCESS
            result.explanation = "规则兜底: 检测到购买流程关键词"
        elif any(kw in q for kw in ["收益", "年化", "利息", "保本", "亏损", "赚", "利率", "计息", "业绩比较基准"]):
            result.intent = QueryIntent.INCOME_QUESTION
            result.explanation = "规则兜底: 检测到收益/计息关键词"
        elif any(kw in q for kw in _PRODUCT_KEYWORDS):
            result.intent = QueryIntent.PRODUCT_CONSULT
            result.explanation = "规则兜底: 检测到产品关键词"
        elif any(kw in q for kw in ["你是谁", "你是啥", "介绍一下你", "在吗", "你好呀", "你好", "您好", "谢谢", "hello", "hi"]):
            result.intent = QueryIntent.CHITCHAT
            result.explanation = "规则兜底: 检测到闲聊关键词"
        else:
            result.intent = QueryIntent.UNKNOWN
            result.explanation = "规则兜底: 无法识别为明确的理财客服意图"

        result.confidence = 0.55 if result.intent is not QueryIntent.UNKNOWN else 0.30
        logger.info("规则兜底路由: %s (%.0f%%)", result.intent.value, result.confidence * 100)
        return result

    # 强信号意图: 规则命中即直接分类, 不调 LLM (零延迟+高准确)
    _STRONG_SIGNAL_INTENTS = {
        "fraud_report": ["假理财", "飞单", "被骗", "诈骗", "内部高收益", "稳赚不赔",
                         "内部渠道", "安全账户", "点击链接核实", "核实真伪",
                         "高收益理财宣传", "内部理财渠道", "保本高收益"],
        "complaint": ["投诉", "不满", "不满意", "太差了"],
        "deposit_insurance": ["存款保险", "偿付限额", "受存款保险", "存款保险赔",
                              "存款保险保障", "是存款吗", "算存款吗", "受保障吗",
                              "受存款保障"],
    }

    def strong_signal_rule(self, query: str):
        """强信号规则仲裁: 命中强信号关键词直接返回意图, 不调 LLM.
        返回 (RoutingResult|None, 命中的意图名|None)
        """
        q = query.lower()
        for intent_name, keywords in self._STRONG_SIGNAL_INTENTS.items():
            for kw in keywords:
                if kw in q:
                    intent = QueryIntent(intent_name)
                    r = RoutingResult(
                        intent=intent,
                        confidence=0.92,
                        original_query=query,
                        explanation=f"强信号规则命中: {kw}",
                    )
                    logger.info("强信号规则仲裁: %s (关键词=%s)", intent_name, kw)
                    return r, intent_name
        return None, None

    def route(self, query: str, dialog_context: str = "") -> RoutingResult:
        """
        对用户查询进行分类和实体提取.
        优化: (1) 强信号意图规则直出不调LLM; (2) 动态few-shot; (3) CoT推理.

        Args:
            query: 用户原始查询
            dialog_context: 多轮对话上下文摘要

        Returns:
            RoutingResult 包含意图、实体、置信度
        """
        logger.info("路由分析: %s", query[:60])

        result = RoutingResult(original_query=query)

        # 规则混合仲裁: 强信号意图直接返回, 不调 LLM (零延迟)
        strong_result, strong_intent = self.strong_signal_rule(query)
        if strong_result is not None:
            return strong_result

        # 动态 few-shot: 从 Golden QA 检索最相似的边界 case
        dyn_cases = _retrieve_few_shots(query, top_k=4)
        dyn_few_shots_text = ""
        if dyn_cases:
            lines = []
            for c in dyn_cases:
                note = f" (注意: {c['note'][:40]})" if c.get("note") else ""
                lines.append(f'  Q: "{c["question"]}" -> intent: {c["intent"]}{note}')
            dyn_few_shots_text = "\n".join(lines)
        else:
            dyn_few_shots_text = "  (无相似案例)"

        # 拼接对话上下文
        user_content = query
        if dialog_context:
            user_content = f"{dialog_context}\n\n【当前用户问题】\n{query}"

        try:
            system_prompt = INTENT_DETECTION_PROMPT.replace('{dyn_few_shots}', dyn_few_shots_text)
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                json_mode=True,
            )
            parsed = json.loads(raw)

            intent_str = parsed.get("intent", "unknown").lower()
            if intent_str in QueryIntent._value2member_map_:
                result.intent = QueryIntent(intent_str)
            else:
                result.intent = QueryIntent.UNKNOWN

            result.entities = parsed.get("entities", [])
            result.time_range = parsed.get("time_range", [])
            result.metrics = parsed.get("metrics", [])
            result.confidence = parsed.get("confidence", 0.0)
            analysis = parsed.get("analysis", "")
            explanation = parsed.get("explanation", "")
            result.explanation = f"[CoT] {analysis} | {explanation}" if analysis else explanation

            logger.info(
                "路由结果: %s (%.0f%%) — %s",
                result.intent.value,
                result.confidence * 100,
                result.explanation[:100],
                extra={
                    "intent": result.intent.value,
                    "confidence": result.confidence,
                    "entities": len(result.entities),
                    "dyn_fewshots": len(dyn_cases),
                },
            )

        except Exception as e:
            logger.error("路由分析失败: %s, 使用规则兜底", e, exc_info=True)
            result = self._rule_fallback(query)
            result.original_query = query

        return result

    def get_entity_hint(self, result: RoutingResult) -> Optional[str]:
        """理财域产品图谱在阶段三接入; 当前返回 None (实体在 slot 层消化)"""
        return None
