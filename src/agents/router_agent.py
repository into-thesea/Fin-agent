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


# 意图检测 Prompt
INTENT_DETECTION_PROMPT = """你是一个银行/金融理财产品智能客服的查询分类器。请分析用户问题，输出 JSON。

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
  chitchat:          闲聊 ("你是谁" / "你好")
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
  "intent": "product_consult|product_compare|risk_suitability|income_question|deposit_insurance|buy_process|hold_redeem|fee_rule|fraud_report|complaint|chitchat|unknown",
  "entities": [{"name": "稳盈添利30天", "type": "Product"}, {"name": "R2", "type": "RiskLevel"}],
  "time_range": [],
  "metrics": [],
  "confidence": 0.95,
  "explanation": "简短理由"
}
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
        if any(kw in q for kw in ["假理财", "飞单", "被骗", "诈骗", "举报", "内部渠道", "内部高收益", "稳赚不赔", "靠谱吗"]) or re.search(r"高收益.{0,3}(可靠|真|假|吗)", q):
            result.intent = QueryIntent.FRAUD_REPORT
            result.explanation = "规则兜底: 检测到疑似诈骗/飞单关键词"
        elif any(kw in q for kw in ["对比", "区别", "哪个好", "哪个合适", "哪个适合", "哪个收益", "哪款好", "推荐", "适合我吗"]) or re.search(r"和.{0,8}(哪个|哪款|比)", q):
            result.intent = QueryIntent.PRODUCT_COMPARE
            result.explanation = "规则兜底: 检测到产品对比/选品关键词"
        elif any(kw in q for kw in ["投诉", "不满", "不满意"]):
            result.intent = QueryIntent.COMPLAINT
            result.explanation = "规则兜底: 检测到投诉关键词"
        elif any(kw in q for kw in ["存款保险", "是存款吗", "是不是存款", "保不保", "受保护", "偿付", "50万", "保障范围"]):
            result.intent = QueryIntent.DEPOSIT_INSURANCE
            result.explanation = "规则兜底: 检测到存款保险/产品属性关键词"
        elif any(kw in q for kw in ["风险等级", "测评", "风险承受", "适当性", "能买", "适合买"]) or _RISK_RE.search(q):
            result.intent = QueryIntent.RISK_SUITABILITY
            result.explanation = "规则兜底: 检测到风险等级/适当性关键词"
        elif any(kw in q for kw in ["费率", "手续费", "费用", "管理费", "赎回费", "申购费"]):
            result.intent = QueryIntent.FEE_RULE
            result.explanation = "规则兜底: 检测到费率关键词"
        elif any(kw in q for kw in ["赎回", "到期", "净值", "撤单", "持仓", "提前支取"]):
            result.intent = QueryIntent.HOLD_REDEEM
            result.explanation = "规则兜底: 检测到赎回/持有关键词"
        elif any(kw in q for kw in ["怎么买", "购买", "申购", "流程", "门槛", "起购", "冷静期", "双录", "操作", "怎么买"]):
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

    def route(self, query: str) -> RoutingResult:
        """
        对用户查询进行分类和实体提取

        Args:
            query: 用户原始查询

        Returns:
            RoutingResult 包含意图、实体、置信度
        """
        logger.info("路由分析: %s", query[:60])

        result = RoutingResult(original_query=query)

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": INTENT_DETECTION_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                json_mode=True,
            )
            parsed = json.loads(raw)

            # 解析意图
            intent_str = parsed.get("intent", "unknown").lower()
            if intent_str in QueryIntent._value2member_map_:
                result.intent = QueryIntent(intent_str)
            else:
                result.intent = QueryIntent.UNKNOWN

            # 解析实体
            result.entities = parsed.get("entities", [])
            result.time_range = parsed.get("time_range", [])
            result.metrics = parsed.get("metrics", [])
            result.confidence = parsed.get("confidence", 0.0)
            result.explanation = parsed.get("explanation", "")

            logger.info(
                "路由结果: %s (%.0f%%) — %s",
                result.intent.value,
                result.confidence * 100,
                result.explanation,
                extra={
                    "intent": result.intent.value,
                    "confidence": result.confidence,
                    "entities": len(result.entities),
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
