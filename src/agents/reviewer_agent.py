"""
ReviewerAgent — 反幻觉验证智能体

功能:
  1. 事实核对: 提取回答中的数值断言，与检索原文交叉验证
  2. 缺失预警: 标记那些在上下文中找不到支撑的断言
  3. 置信度评分: 为每条断言分配 high/medium/low 置信度
  4. 整体质量评分: 回答的完整性和准确性评级

流程:
  Agent 生成回答 → ReviewerAgent 验证 → 通过则返回 → 否则触发重新生成
"""

import os
import json
import re
import logging
from typing import Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from src.llm.llm_client import create_client
from src.retrieval.retriever import format_graph_context

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)


@dataclass
class Claim:
    """单个断言的验证结果"""
    statement: str = ""        # 断言原文
    claim_type: str = ""       # numeric | relational | categorical
    value: str = ""            # 断言中的数值
    supported: bool = False    # 是否在原文中找到支撑
    confidence: str = "low"    # high | medium | low
    source: str = ""           # 支撑来源
    issue: str = ""            # 问题描述 (如果不支持)


@dataclass
class ReviewResult:
    """审核结果"""
    overall_score: float = 0.0        # 0-100 综合评分
    claims: list = field(default_factory=list)  # 所有断言的验证
    supported_claims: int = 0
    unsupported_claims: int = 0
    hallucinations: list = field(default_factory=list)  # 被标记为幻觉的断言
    verdict: str = "pass"            # pass | flag | reject
    suggestion: str = ""             # 改进建议


REVIEW_PROMPT = """你是一个智能客服答案质量审核员。请审核 AI 客服生成的回答。

任务:
1. 从回答中提取所有**事实断言**（含金额、天数、政策条款）
2. 逐条对照【参考原文】判断是否得到支撑
3. 评估回答的整体质量

输出 JSON 格式 (必须严格遵循):
{
  "claims": [
    {
      "statement": "稳盈添利30天业绩比较基准为2.80%-3.60%",
      "claim_type": "numeric",
      "value": "2.80%-3.60%",
      "supported": true,
      "confidence": "high",
      "source": "产品说明书-稳盈添利30天",
      "issue": ""
    }
  ],
  "overall_score": 85,
  "verdict": "pass",
  "suggestion": "无显著问题"
}

要求:
- supported=true 仅当原文明确包含该事实/条款
- confidence: high=原文完全匹配; medium=合理推断但无直接原文; low=无任何支撑
- verdict: pass(通过) / flag(有小问题) / reject(严重幻觉需重生成)
- 如果是 reject, suggestion 必须说明需要修正的具体断言"""


class ReviewerAgent:
    """反幻觉验证 Agent"""

    def __init__(self):
        self.client = create_client(cheap=True)

    def review(self, answer: str, contexts: dict, web_context: str = None) -> ReviewResult:
        """
        审核 AI 回答的质量

        Args:
            answer: AI 生成的回答
            contexts: 三路召回上下文 (与生成时相同)
            web_context: 可选的 Web 搜索结果 (用于交叉验证)

        Returns:
            ReviewResult 包含逐条断言验证和总体评分
        """
        logger.info("ReviewerAgent 开始审核...")

        # 组装上下文原文 (用于 LLM 判断)
        local_text = "\n".join(
            [f"- {c.get('content', '')[:800]} [来源: {c.get('source', '未知')}]"
             for c in contexts.get('local', [])]
        )
        graph_text = format_graph_context(contexts.get('graph', []))
        global_text = "\n".join(
            [f"- 宏观背景: {s}" for s in contexts.get('global', [])]
        )
        web_text = f"\n【网络补充信息】:\n{web_context}" if web_context else ""

        full_context = f"""
【参考原文】:
{local_text}

【宏观背景】:
{global_text}

【知识图谱】:
{graph_text}{web_text}
"""

        try:
            raw = self.client.chat(
                messages=[
                    {"role": "system", "content": REVIEW_PROMPT},
                    {"role": "user", "content": f"【待审核回答】:\n{answer}\n\n【参考上下文】:\n{full_context}"},
                ],
                temperature=0.0,
                json_mode=True,
            )
            parsed = json.loads(raw)

            # 构建 ReviewResult
            result = ReviewResult()
            result.overall_score = parsed.get("overall_score", 50)
            result.verdict = parsed.get("verdict", "pass")
            result.suggestion = parsed.get("suggestion", "")

            for c in parsed.get("claims", []):
                claim = Claim(
                    statement=c.get("statement", ""),
                    claim_type=c.get("claim_type", ""),
                    value=c.get("value", ""),
                    supported=c.get("supported", False),
                    confidence=c.get("confidence", "low"),
                    source=c.get("source", ""),
                    issue=c.get("issue", ""),
                )
                result.claims.append(claim)
                if claim.supported:
                    result.supported_claims += 1
                else:
                    result.unsupported_claims += 1
                    if claim.confidence == "low":
                        result.hallucinations.append(claim.statement)

            logger.info(
                "审核结果: %s (score=%.0f, claims=%d, unsupported=%d)",
                result.verdict,
                result.overall_score,
                len(result.claims),
                result.unsupported_claims,
                extra={
                    "verdict": result.verdict,
                    "score": result.overall_score,
                    "claims": len(result.claims),
                    "unsupported": result.unsupported_claims,
                    "hallucinations": len(result.hallucinations),
                },
            )

            return result

        except Exception as e:
            logger.error("审核失败: %s", e, exc_info=True)
            return ReviewResult(
                overall_score=50,
                verdict="pass",
                suggestion=f"审核过程出错: {e}",
            )

    # ──────────────────────────────────────────────
    # 轻量级本地检查 (作为 LLM 审核的补充)
    # ──────────────────────────────────────────────

    def quick_check(self, answer: str, contexts: dict) -> ReviewResult:
        """
        快速本地检查 (不依赖 LLM):
          1. 数字断言必须出现在检索上下文中
          2. 引用来源必须是检索来源子集
          3. 检索充分却答"未找到" → 检索与问题不匹配
          4. 长回答未引用来源 → 可能未基于检索内容

        作为 LLM 审核的前置过滤，捕获明显幻觉/接地性缺失。
        """
        local = contexts.get('local', [])
        context_text = " ".join([c.get('content', '') for c in local])
        retrieved_sources = {str(c.get('source', '')).strip() for c in local if c.get('source')}

        result = ReviewResult()
        issues = []

        # ── 1. 数字断言: 提取所有"数字+单位"模式, 检查是否在上下文中出现 ──
        number_patterns = re.findall(
            r'([\d,]+\.?\d*)\s*(元|万元|万|亿|折|%|％|天|个月|年|期|件|次|个|小时|分钟|公里|克|公斤|号)',
            answer
        )
        for num, unit in number_patterns:
            if num not in context_text:
                issues.append(f"数字 {num}{unit} 可能在上下文中找不到直接匹配")
                result.unsupported_claims += 1

        # ── 2. 引用来源有效性: 答案引用的来源必须是检索来源子集 ──
        cited = re.findall(r'\[来源[:：]\s*([^\]]+)\]', answer)
        for src in cited:
            s = src.strip()
            if s and s not in retrieved_sources:
                issues.append(f"回答引用了检索来源之外的来源: {s}")

        # ── 3. 检索有货却答"未找到" → 检索与问题不匹配 ──
        not_found_phrases = ["未找到", "无法确定", "不存在该数据", "未在资料"]
        says_not_found = any(p in answer for p in not_found_phrases)
        if says_not_found and len(local) >= 3:
            issues.append("检索到较充分的上下文却回答'未找到'，可能检索与问题不匹配")

        # ── 4. 未引用来源的长回答 → 可能未基于检索内容生成 ──
        has_citation = bool(cited)
        if local and len(answer) > 80 and not has_citation and not says_not_found:
            issues.append("回答较长但未引用任何检索来源，可能未基于检索内容生成")

        result.claims = [Claim(statement=i) for i in issues]
        result.unsupported_claims = sum(1 for i in issues if i.startswith("数字"))
        result.hallucinations = [i for i in issues if i.startswith("数字")]

        if issues:
            result.verdict = "flag"
            result.suggestion = "; ".join(issues[:3])
        else:
            result.verdict = "pass"

        return result
