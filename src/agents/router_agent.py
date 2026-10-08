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

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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
    SERVICE_POLICY = "service_policy"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


class IntentCategory(str, Enum):
    """第一层粗分类 (4 大类, 边界清晰, 几乎不会判错)"""
    PRODUCT_INFO = "product_info"       # 产品咨询类: 问产品是什么/有什么/对比/存款属性
    RETURN_RISK = "return_risk"         # 收益风险类: 问收益口径/保本/适当性
    TRANSACTION = "transaction"         # 交易操作类: 问申购/赎回/费率
    SERVICE_SAFETY = "service_safety"   # 服务安全类: 问诈骗/投诉/服务规则/闲聊


# 大类 → 该大类下的小类
CATEGORY_INTENTS = {
    IntentCategory.PRODUCT_INFO: [
        QueryIntent.PRODUCT_CONSULT,
        QueryIntent.PRODUCT_COMPARE,
        QueryIntent.DEPOSIT_INSURANCE,
    ],
    IntentCategory.RETURN_RISK: [
        QueryIntent.INCOME_QUESTION,
        QueryIntent.RISK_SUITABILITY,
    ],
    IntentCategory.TRANSACTION: [
        QueryIntent.BUY_PROCESS,
        QueryIntent.HOLD_REDEEM,
        QueryIntent.FEE_RULE,
    ],
    IntentCategory.SERVICE_SAFETY: [
        QueryIntent.FRAUD_REPORT,
        QueryIntent.COMPLAINT,
        QueryIntent.SERVICE_POLICY,
        QueryIntent.CHITCHAT,
    ],
}

# 小类 → 所属大类 (反向映射, 供规则直出/兜底时补 category)
INTENT_CATEGORY = {}
for _cat, _intents in CATEGORY_INTENTS.items():
    for _intent in _intents:
        INTENT_CATEGORY[_intent] = _cat


@dataclass
class RoutingResult:
    """路由结果 (层级式: category 大类 + intent 小类)"""
    intent: QueryIntent = QueryIntent.UNKNOWN
    category: Optional[IntentCategory] = None  # 第一层粗分类
    entities: list = field(default_factory=list)
    time_range: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    confidence: float = 0.0
    original_query: str = ""
    explanation: str = ""


# 层级式意图检测 Prompt (两层: 先 4 大类, 再大类内小类; 一次调用输出两层)
INTENT_DETECTION_PROMPT = """你是一个银行/金融理财产品智能客服的查询分类器。请按"先大类、后小类"两层步骤分析用户问题，再输出分类 JSON。

【推理步骤】
请在 analysis 字段中简要写出以下推理:
1. 第一层——大类判断: 用户问题的核心诉求属于哪一大类? (4 选 1, 大类边界清晰)
   - product_info 产品咨询类: 问产品本身是什么、有哪些产品、产品对比、是不是存款/受不受存款保险
   - return_risk 收益风险类: 问收益怎么算、保不保本、净值、业绩基准、我能不能买、风险测评匹配
   - transaction 交易操作类: 问怎么买、申购流程、赎回、到期、提前支取、费率手续费
   - service_safety 服务安全类: 问诈骗举报、投诉、客服服务规则、隐私、问候寒暄、领域外问题
2. 第二层——小类判断: 在该大类包含的小类中, 哪一个最匹配? (2-4 选 1, 只在大类内部比较, 不要考虑其他大类的小类)
3. 最终小类是什么? 置信度多少?

注意: 必须先确定大类, 再在大类内部确定小类。category 和 intent 必须属于同一大类。

【第一层大类定义】
  product_info 产品咨询类:
    用户想了解"产品本身"——产品是什么、有什么属性、有哪些产品可选、几款产品对比、产品是不是存款/受不受存款保险保障。
  return_risk 收益风险类:
    用户想了解"收益和风险"——收益怎么算、保不保本、利息/年化/净值/业绩基准含义、风险等级与我的测评是否匹配、我能不能买某产品。
  transaction 交易操作类:
    用户想了解"怎么操作"——购买流程、起购门槛、双录、赎回、到期处理、提前支取、费率手续费。
  service_safety 服务安全类:
    用户想了解"服务与安全"——疑似诈骗举报、投诉、客服能做什么/不能做什么、隐私与信息索取、工单时效、问候寒暄、不在理财客服范围内的问题。

【第二层小类定义 (按大类分组, 判断小类时只看所属大类内部)】

product_info 产品咨询类下:
  product_consult:   单款产品的基本属性咨询 (期限/发行方/产品类型/封闭期/开放日/风险等级是多少; 起购金额门槛归 buy_process)
                     例: "安心固收90天是什么类型的产品？" "远见成长混合的持仓周期多久？" "这款产品的风险等级是多少？" "远见成长混合基金是自营还是代销的？"
  product_compare:   两款及以上产品对比, 或按条件筛选/推荐产品
                     例: "国债和定期存款哪个利息高？" "有哪些R1的活期理财？" "这三款哪个流动性最好？"
  deposit_insurance: 问产品是不是存款、受不受存款保险保障、存款保险偿付限额
                     例: "货币基金算存款吗？" "存款保险保外币存款吗？" "通知存款受存款保险保障吗？"

return_risk 收益风险类下:
  income_question:   收益口径/是否保本/利息怎么算/净值与业绩基准含义/会不会亏
                     例: "定期存款3年利率是多少？" "这款产品保本吗？" "万份收益是什么意思？"
                         "业绩比较基准4%是不是保证能拿到？" "理财产品净值下跌是不是就亏了？" "纯债基金历史上亏过本金吗？"
  risk_suitability:  风险等级与用户测评的适当性匹配 (我能不能买某产品/R几能买R几), 以及风险等级含义与等级间区别
                     例: "我是保守型能买远见成长吗？" "R1能买R3的产品吗？" "买保险需要做适当性评估吗？" "R1和R5的风险差距在哪里？"

transaction 交易操作类下:
  buy_process:       怎么买/申购流程/起购门槛/开通权限/冷静期/双录
                     例: "第一次在手机银行买基金要开通什么？" "首次购买私募产品需要什么条件？" "私募基金的投资门槛是多少？"
  hold_redeem:       持有/到期/赎回/提前支取/转让/退保/钱退到哪里
                     例: "封闭期内可以赎回吗？" "理财产品赎回后资金回到哪里？" "年金险买了之后能退保吗？" "定期存款可以转让吗？"
  fee_rule:          费率/手续费/管理费/赎回费
                     例: "持有不满7天赎回有惩罚费吗？" "信托产品的管理费怎么收？"

service_safety 服务安全类下:
  fraud_report:      假理财/飞单/疑似被骗/举报/内部高收益渠道
                     例: "陌生人拉我进群推荐理财" "承诺年化15%的稳赚产品能信吗？" "私下推荐的高收益产品可靠吗？"
  complaint:         投诉/不满/要求处理
                     例: "我要投诉网点服务" "对理财产品销售不满怎么反映？"
  service_policy:    客服服务规则 (转人工条件/客服边界/隐私/测评流程/工单时效)
                     例: "如何接通人工客服？" "客服会要求我提供交易密码吗？" "适当性评估有效期多久？" "客户信息会被共享给第三方吗？"
  chitchat:          仅限问候寒暄, 或明确不在理财客服范围内的问题; 不得收纳任何产品/收益/交易/服务政策类业务问题
                     例: "你好" "谢谢" "帮我分析宁德时代股票" "以太坊价格走势" "帮我写遗嘱" "下周A股怎么走"
                     "我把存折号告诉你帮我查余额" "我把网银密码告诉你帮我转账"
                     "利息多少？"(无产品上下文) "那款产品怎么样？"(无上下文无法确定具体产品)

【实体提取规则】
  - Product: 产品名（"稳盈添利30天" / "大额存单" / "安鑫纯债基金"）
  - ProductType: 产品大类（理财/存款/基金/保险）
  - Amount: 金额（"50万" / "50000元"）
  - Term: 期限（"30天" / "90天" / "3年"）
  - RiskLevel: 风险/测评等级（R1~R5）
  - YieldType: 收益口径（业绩比较基准/七日年化/执行利率/预定利率）

【输出格式 (严格 JSON)】
{
  "analysis": "第一层大类: 用户在问...属于X类, 因为...; 第二层小类: 在X类的a/b/c中, ...最匹配, 因为...; 排除了...因为...",
  "category": "product_info|return_risk|transaction|service_safety",
  "intent": "该大类下的某个小类",
  "entities": [{"name": "稳盈添利30天", "type": "Product"}, {"name": "R2", "type": "RiskLevel"}],
  "time_range": [],
  "metrics": [],
  "confidence": 0.95,
  "explanation": "简短理由"
}

【两层判断示例】
  Q: "定期存款的利息是固定的吗？"
    → category: return_risk (问收益保证, 不是问产品属性)
    → intent: income_question (收益口径, 不是存款保险)
  Q: "国债和大额存单哪个更适合保守型？"
    → category: product_info (问产品对比)
    → intent: product_compare (多产品对比, 不是适当性)
  Q: "我测评是R3，能买远见成长混合吗？"
    → category: return_risk (问我能不能买, 核心是适当性)
    → intent: risk_suitability (适当性匹配, 不是产品咨询)
  Q: "买私募基金为什么要合格投资者认证？"
    → category: transaction (问购买流程要求)
    → intent: buy_process (不是服务规则)
  Q: "适当性评估过期了怎么更新？"
    → category: service_safety (问测评流程, 属于客服服务规则)
    → intent: service_policy (不是适当性匹配)
  Q: "定期存款3年利率是多少？"
    → category: return_risk (问利率/收益, 不是问产品属性)
    → intent: income_question (不是产品咨询)
  Q: "安心固收90天封闭期多久？"
    → category: product_info (问产品属性/期限)
    → intent: product_consult (不是购买流程)
  Q: "客服会要求我提供交易密码吗？"
    → category: service_safety (问客服边界)
    → intent: service_policy (不是购买流程)

【相似历史案例参考】
{dyn_few_shots}

【多轮上下文判断规则】
  - 如果上一轮在咨询某款产品, 本轮用"它/这个/那"指代同一款产品继续问, 大类应与上一轮一致, 小类根据本轮问的具体属性判断。
    例: 上一轮"稳盈添利30天收益怎么样"(return_risk/income_question), 本轮"那它的风险等级呢"
        → category: product_info (问产品属性), intent: product_consult
        注意: 本轮问的是"风险等级是多少"这一产品属性, 不是问"我能不能买"的适当性, 所以不是 risk_suitability。
  - 如果本轮明显开启新话题 (突然问存款保险、突然投诉、换一款产品), 以本轮内容为准, 不要被上一轮大类束缚。
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
        result.category = INTENT_CATEGORY.get(result.intent)
        logger.info("规则兜底路由: %s/%s (%.0f%%)",
                    result.category.value if result.category else "-",
                    result.intent.value, result.confidence * 100)
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
                        category=INTENT_CATEGORY.get(intent),
                        confidence=0.92,
                        original_query=query,
                        explanation=f"强信号规则命中: {kw}",
                    )
                    logger.info("强信号规则仲裁: %s/%s (关键词=%s)",
                                r.category.value if r.category else "-", intent_name, kw)
                    return r, intent_name
        return None, None

    def route(self, query: str, dialog_context: str = "", use_few_shot: bool = True) -> RoutingResult:
        """
        对用户查询进行分类和实体提取.
        优化: (1) 强信号意图规则直出不调LLM; (2) 动态few-shot; (3) CoT推理.

        Args:
            query: 用户原始查询
            dialog_context: 多轮对话上下文摘要
            use_few_shot: 是否启用动态 few-shot (评测时设为 False 避免答案泄漏)

        Returns:
            RoutingResult 包含意图、实体、置信度
        """
        logger.info("路由分析: %s", query[:60])

        result = RoutingResult(original_query=query)

        # 规则混合仲裁: 强信号意图直接返回, 不调 LLM (零延迟)
        strong_result, strong_intent = self.strong_signal_rule(query)
        if strong_result is not None:
            return strong_result

        # 动态 few-shot: 从 Golden QA 检索最相似的边界 case (评测时可关闭以避免答案泄漏)
        dyn_cases = _retrieve_few_shots(query, top_k=4) if use_few_shot else []
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

            # 解析大类, 并校验大类与小类是否一致
            category_str = parsed.get("category", "").lower()
            if category_str in IntentCategory._value2member_map_:
                parsed_category = IntentCategory(category_str)
                # 小类属于该大类才采纳, 否则以小类反查的大类为准 (保证两层一致)
                if result.intent in CATEGORY_INTENTS.get(parsed_category, []):
                    result.category = parsed_category
                else:
                    result.category = INTENT_CATEGORY.get(result.intent)
            else:
                result.category = INTENT_CATEGORY.get(result.intent)

            result.entities = parsed.get("entities", [])
            result.time_range = parsed.get("time_range", [])
            result.metrics = parsed.get("metrics", [])
            result.confidence = parsed.get("confidence", 0.0)
            analysis = parsed.get("analysis", "")
            explanation = parsed.get("explanation", "")
            result.explanation = f"[CoT] {analysis} | {explanation}" if analysis else explanation

            logger.info(
                "路由结果: %s/%s (%.0f%%) — %s",
                result.category.value if result.category else "-",
                result.intent.value,
                result.confidence * 100,
                result.explanation[:100],
                extra={
                    "category": result.category.value if result.category else "",
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
