"""快速通道准入判定 (五道闸串联).

做法参考四家的通行模式 —— Amazon Lex 的 nluIntentConfidenceThreshold +
AMAZON.FallbackIntent、Dialogflow CX 的 ML Classification Threshold +
sys.no-match-default、Rasa 的 FallbackClassifier + nlu_fallback、语义路由的相似度阈值。
共同点是「模型有多确定」而非「句子里有没有某个词」，且低确定时升级到更强路径。

本项目按实测做了调整 (105 条 golden, 见 spec §4.1):
  - LLM 自报的 confidence 饱和在 0.95 (93% 样本 >= 0.95), 阈值区分度弱;
  - 有区分度的是检索 top1 分数 (<0.60 区间意图错 37.5%, >=0.70 为 0)。
所以闸④是主闸, 闸③保留但可关闭 (CONFIDENCE_THRESHOLD = None)。

**换 router 模型后必须重跑 scripts/calibrate_fast_path.py 重标阈值** —— 阈值不跨模型迁移。
设计见 docs/superpowers/specs/2026-10-03-fast-path-confidence-gate-design.md
"""

import logging

logger = logging.getLogger(__name__)

# 答案来自知识库的 7 类意图 (在 INTENT_NODE_MAP 里都落到 finance_node)
INFORMATIONAL_INTENTS = frozenset({
    "product_consult", "product_compare", "risk_suitability", "income_question",
    "deposit_insurance", "fee_rule", "service_policy",
})

# 必须确定性转人工的两类 (cs_graph.HANDOFF_INTENTS 同源)
HANDOFF_INTENTS = frozenset({"fraud_report", "complaint"})

# 答案在用户私有数据里, KB 永远没有 —— 这是结构性判断, 不是难度判断。
# 例: 「我的风险测评等级是多少」意图置信度很高, 但 KB 答不了。
PRIVATE_DATA_MARKERS = ("我的", "查一下", "查查", "查询", "持仓", "我买", "我持有")

CONFIDENCE_THRESHOLD = 0.90   # 闸③; None = 关闭
KB_SCORE_THRESHOLD = 0.60     # 闸④


def top1_score(contexts: dict) -> float:
    """检索融合分的最高分; 无结果/形状异常一律 0.0 (fail closed)"""
    local = (contexts or {}).get("local") or []
    if not local:
        return 0.0
    try:
        return float(local[0].get("score", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def evaluate(intent: str, confidence: float, contexts: dict, query: str,
             strong_intent: str = None) -> tuple:
    """五道闸串联 → (是否走快答, 未过的闸名); 全过时闸名为 "pass"."""
    if strong_intent in HANDOFF_INTENTS:
        return False, "handoff"
    if any(m in (query or "") for m in PRIVATE_DATA_MARKERS):
        return False, "private_data"
    if intent not in INFORMATIONAL_INTENTS:
        return False, "intent_not_informational"
    if CONFIDENCE_THRESHOLD is not None and (confidence or 0.0) < CONFIDENCE_THRESHOLD:
        return False, "low_confidence"
    if top1_score(contexts) < KB_SCORE_THRESHOLD:
        return False, "no_kb_evidence"
    return True, "pass"


def should_answer_fast(intent: str, confidence: float, contexts: dict, query: str,
                       strong_intent: str = None) -> bool:
    return evaluate(intent, confidence, contexts, query, strong_intent)[0]
