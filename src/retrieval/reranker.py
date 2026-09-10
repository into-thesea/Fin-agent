"""
轻量重排器 (SparseBoost) — 零外部模型依赖

在混合检索 (RRF 融合) 之后, 对 top-N 候选按
"查询关键信息与 chunk 的匹配度 + 归一化原始相关分" 加权重排。

信号:
  - 重叠率: query 中的年份 / 公司名 / 财务指标词在 chunk 内容中出现的命中比例
  - 原始相关分: RRF 融合后的 rrf_score (或向量 score), 归一化到 0-1

离线环境无法下载 cross-encoder, 该重排器用稀疏匹配信号近似精排,
对"公司+指标+年份"型查询 (本项目评测集主流) 提升显著。

用法:
    from src.retrieval.reranker import SparseReranker
    reranker = SparseReranker()
    top = reranker.rerank(query, candidates, top_k=5)
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

YEAR_RE = re.compile(r"(?:20\d{2}|19\d{2})")
# 仅英文词/词组 (中文交给 _split_zh 按停用词切分)
TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.\- ]{1,}")

# 中文关键词片段长度上限 (过长片段难以整串命中, 丢弃)
MAX_TERM_LEN = 8

# 高频疑问/停用词, 不参与重叠打分 (长词在前, 用于切分中文)
STOPWORDS = {
    "是多少", "有多少", "怎么样", "什么样", "为什么", "是什么", "有没有",
    "大概", "大约", "请问", "能不能", "分别", "时候", "哪些", "哪个",
    "多少", "什么", "怎么", "如何", "是否", "哪里", "哪种",
    "是", "的", "了", "吗", "呢", "在", "与", "及", "和", "或", "对",
    "为", "从", "到", "就", "请", "介绍", "了解", "情况", "相关",
    "主要", "一下", "那", "这个", "那个", "进行", "其中", "以及",
    "属于", "位于", "期", "公司", "今年", "去年", "本期",
}


class SparseReranker:
    """稀疏信号重排器: overlap 主导 + 原始分辅助"""

    def __init__(self, lambd_overlap: float = 0.6) -> None:
        """
        Args:
            lambd_overlap: 重叠率权重 (0.6 = 重叠主导, 0.4 = 原始分)
        """
        self.lambd_overlap = lambd_overlap

    # ──────────────────────────────────────────────
    # 特征提取
    # ──────────────────────────────────────────────

    @staticmethod
    def extract_features(query: str) -> tuple:
        """提取查询的年份集合与关键词集合 (英文按空格分词, 中文按停用词切分)"""
        years = set(YEAR_RE.findall(query))
        terms = set()
        # 英文 token (英文有空格, 天然分词)
        for t in TERM_RE.findall(query):
            t = t.strip().lower()
            if len(t) >= 2 and not t.isdigit():
                terms.add(t)
        # 中文: 用停用词作分隔符切出有意义片段 (免分词器依赖)
        for part in SparseReranker._split_zh(query):
            terms.add(part)
        return years, terms

    @staticmethod
    def _split_zh(text: str) -> List[str]:
        """切分中文为有意义片段: 先按数字断开, 再按停用词切分 (免分词器)

        例: 比亚迪在2024的营业收入是多少 → 比亚迪 / 营业收入
            宁德时代2024年度与关联方发生... → 宁德时代 / 年度 / 关联方发生
        """
        sw = sorted(STOPWORDS, key=len, reverse=True)
        sw_pattern = "|".join(re.escape(s) for s in sw)
        out = []
        # 1) 按数字断开 (宁德时代2024年度 → 宁德时代 / 年度)
        for seg in re.split(r"\d+", text):
            # 2) 按停用词切分
            for part in re.split(sw_pattern, seg):
                p = part.strip().strip("？?。！!；;：:、，, ")
                if (len(p) >= 2 and len(p) <= MAX_TERM_LEN
                        and re.search(r"[一-鿿]", p)):
                    out.append(p)
        return out

    # ──────────────────────────────────────────────
    # 打分与重排
    # ──────────────────────────────────────────────

    def _overlap_ratio(self, years: set, terms: set, content: str) -> float:
        """查询关键信息在 chunk 中的命中比例"""
        content_l = content.lower()
        year_hits = sum(1 for y in years if y in content_l)
        term_hits = sum(1 for t in terms if t in content_l)
        total = len(years) + len(terms)
        return (year_hits + term_hits) / max(total, 1)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5,
        keep_rrf: Optional[bool] = False,
    ) -> List[dict]:
        """
        对候选重排, 返回 top_k 条

        Args:
            query: 原始查询
            candidates: 混合检索的 top-N 候选 (含 rrf_score/score/content)
            top_k: 返回条数
            keep_rrf: True 时保留原始 rrf 排序作为 tie-breaker
        """
        if len(candidates) <= top_k:
            return candidates
        years, terms = self.extract_features(query)

        # 池内最大原始分 (归一化基准)
        base_scores = [
            c.get("rrf_score") or c.get("score") or 0.0 for c in candidates
        ]
        base_max = max(base_scores) if base_scores else 0.0

        scored = []
        for c in candidates:
            overlap = self._overlap_ratio(years, terms, c.get("content") or "")
            base = c.get("rrf_score") or c.get("score") or 0.0
            norm_base = min(base / base_max, 1.0) if base_max > 0 else 0.0
            score = self.lambd_overlap * overlap + (1 - self.lambd_overlap) * norm_base
            # 全 0 的候选垫底
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        if keep_rrf:
            scored.sort(key=lambda x: (x[0], -x[1].get("rrf_score", 0)), reverse=True)
        return [c for _, c in scored[:top_k]]


class CrossEncoderReranker:
    """CrossEncoder 精排 (bge-reranker-base) — 需本地模型文件

    query 与 doc 拼接进同一 transformer, 可建模 token 级交互 (同义/指代),
    精度高于 SparseBoost, 但每个 query×candidate 都要一次推理, 成本更高。

    用法:
        from src.retrieval.reranker import CrossEncoderReranker
        rr = CrossEncoderReranker("models/bge-reranker-base")
        top = rr.rerank(query, candidates, top_k=5)
    """

    def __init__(self, model_path: str = "models/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CrossEncoder 模型不存在: {model_path}\n"
                f"请先在有网环境下载 BAAI/bge-reranker-base 放入该目录"
            )
        self.model = CrossEncoder(model_path)

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5,
               batch_size: int = 128) -> List[dict]:
        if not candidates:
            return []
        pairs = [(query, c.get("content") or "") for c in candidates]
        # 增大 batch 减少模型调用开销 (CPU 大 batch 也更快)
        scores = self.model.predict(pairs, batch_size=batch_size)
        ordered = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ordered[:top_k]]
