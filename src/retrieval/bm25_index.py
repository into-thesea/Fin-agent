"""
轻量级 BM25 检索实现 (纯 Python, 无外部依赖)

用于密集向量检索的稀疏检索补充。
通过 Reciprocol Rank Fusion (RRF) 合并两种检索结果。
"""

from __future__ import annotations

import math
import os
import pickle
import logging
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25Okapi 检索器

    用法:
        index = BM25Index()
        index.build(documents)        # documents = [{"id": str, "text": str}, ...]
        results = index.search("查询词", top_k=5)
        index.save("bm25_index.pkl")
        index.load("bm25_index.pkl")
    """

    def __init__(self) -> None:
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_sources: List[str] = []
        self.doc_sections: List[str] = []
        self.avgdl = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self.epsilon = 0.25
        self.idf: Optional[dict] = None
        self.doc_freqs: Optional[List[Counter]] = None
        self.nd = 0  # 文档总数

    def _tokenize(self, text: str) -> List[str]:
        """中英混合分词: 中文按字符 bigram, 英文/数字按整词

        原实现用 re.findall(r'[一-鿿\\w]+'), 它把一整串连续中文当成**一个** token:
          "存款保险"        → ['存款保险']       (碰巧能命中)
          "存款保险怎么赔"   → ['存款保险怎么赔']  (永远命中不了)
        而自然中文问句没有空格, 于是 BM25 对绝大多数真实问句直接返回 0 条 ——
        所谓"向量+BM25 混合检索"实际只有向量一路在工作。

        中文按 bigram 切是中文检索的标准免依赖做法 (jieba 未安装, 也不值得为此加依赖)。
        """
        import re
        tokens: List[str] = []
        for run in re.findall(r'[一-鿿]+|[a-zA-Z0-9_]+', text.lower()):
            if run[0].isascii() or len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        return tokens

    def build(self, documents: List[dict]) -> None:
        """
        构建 BM25 索引

        Args:
            documents: [{"id": str, "text": str, ...}, ...]
        """
        # 优先取 chunk_id (jsonl 唯一标识), 兼容旧版 "id" 字段
        self.doc_ids = [d.get("chunk_id") or d.get("id") or str(i) for i, d in enumerate(documents)]
        self.doc_texts = [d.get("content", d.get("text", "")) for d in documents]
        self.doc_sources = [d.get("source", "") for d in documents]
        self.doc_sections = [d.get("section", "") for d in documents]
        self.nd = len(self.doc_texts)

        # 分词并统计词频
        self.doc_freqs = []
        df = Counter()  # 文档频率

        for text in self.doc_texts:
            tokens = self._tokenize(text)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            for term in freq:
                df[term] += 1

        # 计算 IDF
        self.idf = {}
        for term, doc_count in df.items():
            idf = math.log(1 + (self.nd - doc_count + 0.5) / (doc_count + 0.5))
            self.idf[term] = idf

        # 平均文档长度
        total_len = sum(len(f) for f in self.doc_freqs)
        self.avgdl = total_len / self.nd if self.nd > 0 else 0.0

        logger.info("BM25 索引构建完成: %d 文档, %d 不重复词", self.nd, len(self.idf))

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        BM25 检索

        Returns:
            [{"id": str, "content": str, "source": str, "score": float, "chunk_id": str}, ...]
        """
        if not self.idf:
            logger.warning("BM25 索引为空")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i, freq in enumerate(self.doc_freqs):
            score = self._calc_bm25_score(query_tokens, freq)
            if score > 0:
                scores.append((i, score))

        # 按 BM25 分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_k]

        results = []
        for doc_idx, score in scores:
            # source 必须是真实文件名, 不能拿 chunk_id 顶替:
            # 下游 (RRF 融合 / 证据覆盖评测 / 前端来源标注) 都按 source 文件名对齐,
            # 之前这里返回 chunk_id, 导致 BM25 的结果在融合与评测里全部对不上。
            source = (self.doc_sources[doc_idx] if doc_idx < len(self.doc_sources)
                      else self.doc_ids[doc_idx])
            results.append({
                "chunk_id": self.doc_ids[doc_idx],
                "content": self.doc_texts[doc_idx],
                "source": source,
                "section": (self.doc_sections[doc_idx]
                            if doc_idx < len(self.doc_sections) else ""),
                "score": float(score),
            })

        return results

    def _calc_bm25_score(self, query_tokens: List[str], doc_freq: Counter) -> float:
        """计算单个文档的 BM25 分数"""
        score = 0.0
        doc_len = sum(doc_freq.values())

        for term in set(query_tokens):
            if term not in self.idf:
                continue
            tf = doc_freq.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf[term]
            # BM25 公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def save(self, path: str) -> None:
        """持久化到磁盘 (JSON 格式, 向后兼容 pickle)"""
        import json
        json_path = path.replace(".pkl", ".json")

        # 将 Counter 列表转为普通 dict 列表以便 JSON 序列化
        doc_freqs_list = [dict(freq) for freq in self.doc_freqs] if self.doc_freqs else []

        data = {
            "version": "3.1",
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "doc_sources": self.doc_sources,
            "doc_sections": self.doc_sections,
            "avgdl": self.avgdl,
            "avg_doc_length": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "idf": self.idf or {},
            "doc_freqs": doc_freqs_list,
            "nd": self.nd,
            "total_docs": self.nd,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("BM25 索引已保存 (JSON): %s (%d docs)", json_path, self.nd)

        # 同时也写 pickle 格式保持向后兼容
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 索引已保存 (pickle): %s (%d 文档, %.1f KB)",
                     path, self.nd, _file_size(path))

    def load(self, path: str) -> None:
        """从磁盘加载 (优先 JSON, 回退 pickle)"""
        import json
        json_path = path.replace(".pkl", ".json")

        data = None

        # 优先 JSON 格式
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                version = data.get("version", "1.0")
                logger.info("BM25 索引已加载 (JSON v%s): %s", version, json_path)
            except Exception as e:
                logger.warning("BM25 JSON 加载失败, 回退 pickle: %s", e)
                data = None

        # 回退 pickle 格式
        if data is None:
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                logger.info("BM25 索引已加载 (pickle): %s (%d 文档)", path, data.get("nd", data.get("total_docs", 0)))
            except Exception as e:
                logger.warning("BM25 索引加载失败: %s", e)
                self.doc_ids = []
                self.doc_texts = []
                self.doc_sources = []
                self.doc_sections = []
                self.doc_freqs = []
                self.idf = {}
                self.nd = 0
                self.avgdl = 0.0
                self.k1 = 1.5
                self.b = 0.75
                return

        self.doc_ids = data.get("doc_ids", [])
        self.doc_texts = data.get("doc_texts", [])
        self.doc_sources = data.get("doc_sources", [])     # v3.0 起才有
        self.doc_sections = data.get("doc_sections", [])   # v3.1 起才有
        self.avgdl = data.get("avgdl", data.get("avg_doc_length", 0.0))
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        self.idf = data.get("idf", {})
        self.nd = data.get("nd", data.get("total_docs", len(self.doc_ids)))

        # 恢复 Counter 列表 (JSON 存储为普通 dict 列表)
        raw_freqs = data.get("doc_freqs", [])
        if raw_freqs and isinstance(raw_freqs[0], dict):
            self.doc_freqs = [Counter(f) for f in raw_freqs]
        elif raw_freqs and isinstance(raw_freqs[0], Counter):
            self.doc_freqs = raw_freqs
        else:
            self.doc_freqs = []


def rrf_fusion(dense_results: List[dict], sparse_results: List[dict],
                top_k: int = 5, k: int = 60) -> List[dict]:
    """
    Reciprocol Rank Fusion — 合并密集检索和稀疏检索结果

    Args:
        dense_results: 密集向量检索结果 [{"chunk_id": str, "content": str, "source": str, "score": float}]
        sparse_results: BM25 检索结果 [{"chunk_id": str, ...}]
        top_k: 最终返回条数
        k: RRF 常数 (通常 60)

    Returns:
        合并排序后的结果列表
    """
    rrf_scores = {}
    seen = {}

    # 密集检索
    for rank, doc in enumerate(dense_results):
        cid = doc.get("chunk_id", doc.get("content", "")[:80])
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        seen[cid] = doc

    # 稀疏检索
    for rank, doc in enumerate(sparse_results):
        cid = doc.get("chunk_id", doc.get("content", "")[:80])
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in seen:
            seen[cid] = doc

    # 按 RRF 分数排序
    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
    sorted_cids = sorted_cids[:top_k]

    results = []
    for cid in sorted_cids:
        doc = dict(seen[cid])
        doc["rrf_score"] = rrf_scores[cid]
        results.append(doc)

    return results


def rebuild_bm25_index(chunks_path: str, output_path: str) -> BM25Index:
    """
    从 chunks_processed.jsonl 重建 BM25 索引

    Args:
        chunks_path: chunks_processed.jsonl 路径
        output_path: bm25_index.pkl 输出路径
    """
    import json

    all_documents = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                all_documents.append(doc)
    # 父子分块: BM25只索引子块 (父块是完整章节, 不直接检索)
    documents = [d for d in all_documents if d.get("chunk_type") != "parent"]
    logger.info("BM25索引: %d 子块 (跳过 %d 父块)", len(documents), len(all_documents) - len(documents))

    index = BM25Index()
    index.build(documents)
    index.save(output_path)
    return index


def _file_size(path: str) -> float:
    import os
    return os.path.getsize(path) / 1024
