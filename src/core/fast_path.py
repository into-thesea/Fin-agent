"""快速通道准入判定 (四道闸串联).

做法参考四家的通行模式 —— Amazon Lex 的 nluIntentConfidenceThreshold +
AMAZON.FallbackIntent、Dialogflow CX 的 ML Classification Threshold +
sys.no-match-default、Rasa 的 FallbackClassifier + nlu_fallback、语义路由的相似度阈值。
共同点是「模型有多确定」而非「句子里有没有某个词」，且低确定时升级到更强路径。

**与四家的差异 (实测得出, 非偏好)**: 那四家都有一道置信度阈值闸, 本项目**没有** ——
router 自报的 confidence 饱和在 0.95 (105 条 golden 里 93% 在 0.95 以上, 0.95 档里错 6 条、
低于 0.95 的 4 条里错 3 条: 方向对但没区分度)。阈值从 0.70 扫到 0.85 结果一字不差,
0.90/0.95/0.98 只减覆盖不减错误。全链路实测: 关闭 = 0.905、T=0.90 = 0.895, 差 1.0pp
落在 ±2pp 抖动带内, 按事先定死的规则判无差异, 于是**整道闸被移除而不是留一个恒真的开关**。
有区分度的是检索 top1 分数 (<0.60 区间意图错 37.5%, >=0.70 为 0)。
详见 docs/superpowers/specs/2026-10-03-fast-path-confidence-gate-design.md §4

**换 router 模型后必须重跑 scripts/calibrate_fast_path.py 重标 KB_SCORE_THRESHOLD** ——
阈值不跨模型迁移。若换了校准更好的分类器, 置信度闸可以重新加回来。
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

KB_SCORE_THRESHOLD = 0.60     # 闸④, 主闸


def top1_score(contexts: dict) -> float:
    """检索融合分的最高分; 无结果/形状异常一律 0.0 (fail closed)"""
    local = (contexts or {}).get("local") or []
    if not local:
        return 0.0
    try:
        return float(local[0].get("score", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def evaluate(intent: str, contexts: dict, query: str,
             strong_intent: str = None) -> tuple:
    """四道闸串联 → (是否走快答, 未过的闸名); 全过时闸名为 "pass"."""
    if strong_intent in HANDOFF_INTENTS:
        return False, "handoff"
    if any(m in (query or "") for m in PRIVATE_DATA_MARKERS):
        return False, "private_data"
    if intent not in INFORMATIONAL_INTENTS:
        return False, "intent_not_informational"
    if top1_score(contexts) < KB_SCORE_THRESHOLD:
        return False, "no_kb_evidence"
    return True, "pass"


def should_answer_fast(intent: str, contexts: dict, query: str,
                       strong_intent: str = None) -> bool:
    return evaluate(intent, contexts, query, strong_intent)[0]


def route_class(query: str, strong_intent: str = None) -> str:
    """查询分类的**唯一入口**: 规则分类 + 硬前置拦截.

    greeting 优先判定 (greeting 不查 KB、不需意图, 不受其他闸影响);
    命中 HANDOFF 强信号或含私有数据措辞一律强制走完整管道。
    """
    from src.llm.query_router import classify as kw_classify
    qclass = kw_classify(query)
    if qclass == "greeting":
        return "greeting"
    if strong_intent in HANDOFF_INTENTS:
        logger.info("硬前置: 诈骗/投诉强信号 → 强制完整管道")
        return "complex"
    if any(m in (query or "") for m in PRIVATE_DATA_MARKERS):
        logger.info("硬前置: 私有数据措辞 → 强制完整管道")
        return "complex"
    return qclass
