"""
Query Decomposer — LLM 多问题拆分

将用户一次输入的多个独立问题拆成子问题列表，分别检索后合并上下文。
用于处理"这个基金收益怎么算？风险大吗？适合我买吗？"这类多问题输入。

设计:
  - LLM 结构化输出 JSON 数组，每个元素是一个可独立检索的子问题
  - 单问题时返回 [原问题]，不增加额外检索
  - LLM 调用失败/超时时降级返回 [原问题]，不阻断主流程
  - 子问题数量上限 5，避免过度拆分导致延迟爆炸
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

MAX_SUB_QUERIES = 5

_DECOMPOSE_PROMPT = """你是一个金融理财产品客服的问题拆分器。

请将用户输入拆分成若干个**可独立检索答案**的子问题。

规则:
1. 如果用户只问了一个问题，返回只含原问题的数组
2. 如果用户问了多个独立问题（用问号、分号、"还有"、"另外"、"以及"等连接），拆成多个子问题
3. 每个子问题必须语义完整、可独立检索，不要保留代词（"它""这个"）——如果原问题有指代，保留原样即可，检索层会做 Query Rewriting
4. 不要拆分一个完整问题（如"收益和风险怎么样"是一个问题，不要拆成"收益怎么样"+"风险怎么样"）
5. 最多拆成 {max} 个子问题

用户输入: {query}

请严格以 JSON 数组格式输出，例如:
["稳盈添利30天的收益怎么算？", "稳盈添利30天的风险等级是什么？"]
"""


def decompose_query(query: str, max_sub: int = MAX_SUB_QUERIES) -> list[str]:
    """用 LLM 将用户输入拆成子问题列表。

    Args:
        query: 用户原始输入
        max_sub: 子问题数量上限

    Returns:
        子问题列表。单问题时返回 [query]。LLM 失败时降级返回 [query]。
    """
    # 快速路径: 没有问号/分号/连接词，大概率单问题，跳过 LLM 调用省延迟
    if not any(sep in query for sep in ["？", "?", "；", ";", "还有", "另外", "以及", "，再"]):
        return [query]

    try:
        from src.llm.llm_client import create_client
        client = create_client(cheap=True)
        prompt = _DECOMPOSE_PROMPT.format(query=query, max=max_sub)
        raw = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        # 提取 JSON 数组（LLM 可能输出 ```json 包裹或前后有文字）
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            logger.warning("decompose 未提取到 JSON，降级: %s", raw[:100])
            return [query]
        sub_queries = json.loads(match.group())
        # 校验: 必须是非空字符串列表
        if not isinstance(sub_queries, list) or not sub_queries:
            return [query]
        sub_queries = [s.strip() for s in sub_queries if isinstance(s, str) and s.strip()]
        if not sub_queries:
            return [query]
        # 去重 + 截断上限
        seen = set()
        unique = []
        for q in sub_queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        result = unique[:max_sub]
        if len(result) > 1:
            logger.info("decompose: 拆成 %d 个子问题", len(result))
        return result
    except Exception as e:
        logger.warning("decompose LLM 调用失败，降级为单问题: %s", e)
        return [query]
