"""
重排器 —— 三种实现, 用途不同, 别选错:

  ApiReranker (gte-rerank-v2)     ← **线上该用这个**
      query×doc 语义交互模型, 提供第一级(RRF 向量+BM25)没有的信号。
  CrossEncoderReranker (本地 CE)
      同类信号, 需本地模型文件; 本环境 HF 离线且未下载, 实际不可用。
  SparseReranker (SparseBoost)    ← 自研词面启发式, **实测负收益, 不要接线**
      信号是 BM25 的子集而 RRF 里已经有 BM25; 且以 0.6 权重「替换」而非融合已排好的
      次序。理财 golden 实测 Recall@1 82.5%→75.0%, MRR 0.8938→0.8375。
      留在文件里只为记录这个结论, 不是为了启用。

共同接口: rerank(query, candidates, top_k) -> list[dict]

--- 原 SparseBoost 说明 ---

在混合检索 (RRF 融合) 之后, 对 top-N 候选按
"查询关键信息与 chunk 的匹配度 + 归一化原始相关分" 加权重排。

信号 (理财域):
  - 产品名: 与 catalog.jsonl (KNOWN_PRODUCTS) 对齐的产品全名/口语别名
  - 风险等级: R1~R5 与低/中低/中/中高/高风险
  - 期限与数字: 30天 / 90天 / 3年 这类量词
  - 条款术语: 存款保险 / 业绩比较基准 / 犹豫期 / 适当性 / 赎回 …
  - 原始相关分: RRF 融合后的 rrf_score (或向量 score), 归一化到 0-1

注意: 本类原先用的是**文档问答域**词表 (年份/公司名/财务指标), 在理财客服域实测
是负收益 (Recall@1 82.5% → 75.0%, MRR 0.8938 → 0.8375)。现改为域适配词表,
仍须用 scripts/eval_retrieval.py 做 A/B 验证后再决定是否接进线上。

用法:
    from src.retrieval.reranker import SparseReranker
    reranker = SparseReranker()
    top = reranker.rerank(query, candidates, top_k=5)
"""

from __future__ import annotations

import os
import re
import json
import logging
import urllib.request
import urllib.error
from typing import List, Optional

logger = logging.getLogger(__name__)

YEAR_RE = re.compile(r"(?:20\d{2}|19\d{2})")
# 仅英文词/词组 (中文交给 _split_zh 按停用词切分)
TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.\- ]{1,}")
# 风险等级 (R1~R5 与中文档位)
RISK_RE = re.compile(r"R[1-5]|低风险|中低风险|中风险|中高风险|高风险")
# 期限/收益量词 (30天 / 90天 / 3年 / 2.10%)
NUMWORD_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:天|个月|年|年化|万元|元|%)?")

# 理财域条款/流程/合规关键概念 —— 命中即强信号
CORE_TERMS = (
    "存款保险", "非存款", "不受存款保险", "保本", "保收益", "业绩比较基准",
    "七日年化", "万份收益", "赎回", "申购", "冷静期", "犹豫期", "双录",
    "适当性", "风险等级", "风险承受", "起购", "到账", "确认", "费率",
    "管理费", "托管费", "销售服务费", "大额存单", "结构性存款", "定期",
    "活期", "封闭期", "开放日", "提前支取", "违约", "征信", "退保",
    "养老金", "增额终身寿", "代销", "自营", "净值", "T+0", "T+1",
)

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
    def _match_products(query: str) -> List[str]:
        """匹配查询中的产品名 (含口语别名), 长名优先去重。

        产品表来自 catalog.jsonl 单一事实源 (slot_filler.KNOWN_PRODUCTS),
        别名复用图谱检索器那份 —— 同一份领域词表, 不重复维护。
        """
        try:
            from src.core.slot_filler import KNOWN_PRODUCTS
            from src.knowledge_graph.retriever import _PRODUCT_ALIASES
        except Exception:
            return []
        cands = set()
        for full in KNOWN_PRODUCTS:
            cands.add(full)
            cands.update(_PRODUCT_ALIASES.get(full, []))
        hits: List[str] = []
        for c in sorted(cands, key=len, reverse=True):
            if c in query and not any(c != o and c in o for o in hits):
                hits.append(c)
        return hits

    @staticmethod
    def extract_features(query: str) -> tuple:
        """提取 (强信号集合, 关键词集合)

        强信号 = 产品名 / 风险等级 / 期限与数字; 关键词 = 条款术语 + 停用词切分片段。
        比原先的「年份 + 英文公司名」更贴合理财问答。
        """
        strong = set(SparseReranker._match_products(query))
        strong |= set(RISK_RE.findall(query))
        strong |= {m.strip() for m in NUMWORD_RE.findall(query) if m.strip()}

        terms = {t for t in CORE_TERMS if t in query}
        # 英文 token (英文有空格, 天然分词)
        for t in TERM_RE.findall(query):
            t = t.strip().lower()
            if len(t) >= 2 and not t.isdigit():
                terms.add(t)
        # 中文兜底: 用停用词作分隔符切出有意义片段 (免分词器依赖)
        for part in SparseReranker._split_zh(query):
            terms.add(part)
        return strong, terms

    @staticmethod
    def _split_zh(text: str) -> List[str]:
        """切分中文为有意义片段: 先按数字断开, 再按停用词切分 (免分词器)

        例: 稳盈添利30天2024年的收益率是多少 → 稳盈添利 / 天 / 收益率
            大额存单3年期与定期存款的区别 → 大额存单 / 年期 / 定期存款的区别
        """
        sw = sorted(STOPWORDS, key=len, reverse=True)
        sw_pattern = "|".join(re.escape(s) for s in sw)
        out = []
        # 1) 按数字断开 (大额存单3年期 → 大额存单 / 年期)
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

    def _overlap_ratio(self, strong: set, terms: set, content: str) -> float:
        """查询关键信息在 chunk 中的命中比例"""
        content_l = content.lower()
        strong_hits = sum(1 for y in strong if y.lower() in content_l)
        term_hits = sum(1 for t in terms if t.lower() in content_l)
        total = len(strong) + len(terms)
        return (strong_hits + term_hits) / max(total, 1)

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
        strong, terms = self.extract_features(query)

        # 池内最大原始分 (归一化基准)
        base_scores = [
            c.get("rrf_score") or c.get("score") or 0.0 for c in candidates
        ]
        base_max = max(base_scores) if base_scores else 0.0

        scored = []
        for c in candidates:
            overlap = self._overlap_ratio(strong, terms, c.get("content") or "")
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


class ApiReranker:
    """阿里云百炼 gte-rerank-v2 语义精排 —— 线上推荐使用

    与 SparseBoost 的根本区别在于**信号是否与第一级重复**:
      - SparseBoost 算的是词面重合, 而 RRF 的稀疏路已经是 BM25(带 IDF 加权),
        等于把同一类信号用更粗的算法重算一遍 —— 实测负收益。
      - 本类是 query×doc 交互的语义相关性模型, 能识别同义/改写/指代,
        提供向量与 BM25 都给不出的判断依据。

    只要文本进得去就出得来, 无需本地模型文件, 因此在本机(离线 HF)是唯一可用的精排路径。
    """

    ENDPOINT = ("https://dashscope.aliyuncs.com/api/v1/services/"
                "rerank/text-rerank/text-rerank")

    def __init__(self, model: str = "gte-rerank-v2", api_key: Optional[str] = None,
                 timeout: float = 20.0) -> None:
        from src.config import settings
        self.model = model
        self.api_key = api_key or settings.dashscope_api_key
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "未配置 DASHSCOPE_API_KEY —— 无法使用 API 重排。"
                "请在 .env 填好, 或把 RERANK_ENABLED 关掉。"
            )

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """按 query 与各候选的语义相关性重排, 返回 top_k 条。

        失败时抛异常 —— 由调用方决定如何处置; 不在这里静默返回原始顺序,
        否则「重排没生效」会和「重排后顺序恰好没变」无法区分。
        """
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates[:top_k]

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": [c.get("content") or "" for c in candidates],
            },
            "parameters": {"return_documents": False, "top_n": len(candidates)},
        }
        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"重排接口 HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"重排接口调用失败: {e}") from e

        results = ((data.get("output") or {}).get("results")) or []
        if not results:
            raise RuntimeError(f"重排接口返回空结果: {str(data)[:200]}")

        ordered = sorted(results, key=lambda r: r.get("relevance_score") or 0.0,
                         reverse=True)
        out = []
        for r in ordered[:top_k]:
            idx = r.get("index")
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                item = dict(candidates[idx])
                item["rerank_score"] = float(r.get("relevance_score") or 0.0)
                out.append(item)
        if not out:
            raise RuntimeError("重排接口返回的下标全部越界")
        return out
