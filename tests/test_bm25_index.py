"""BM25 分词与 source 字段 — 两个曾让稀疏检索整体失效的 bug

bug 1: _tokenize 把一整串连续中文当成一个 token, 于是"存款保险怎么赔"变成
       单个 token, 永远命中不了任何文档。自然中文问句没有空格 → BM25 基本全废。
bug 2: search() 把 chunk_id 当 source 返回, 下游 (RRF 融合 / 证据覆盖评测 /
       前端来源标注) 全按 source 文件名对齐, 于是对不上。
       而 bug 1 让 BM25 返回 0 条, hybrid_retrieve 静默降级成纯 dense,
       正好把 bug 2 掩盖了 —— 三个 bug 叠在一起, "混合检索"其实只有一路。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.retrieval.bm25_index import BM25Index


def _index():
    idx = BM25Index()
    idx.build([
        {"chunk_id": "c1", "source": "a.md", "section": "s1",
         "content": "存款保险保障存款类产品，单家银行五十万元以内全额偿付。"},
        {"chunk_id": "c2", "source": "b.md", "section": "s2",
         "content": "理财非存款，不受存款保险保障，不承诺保本保收益。"},
        {"chunk_id": "c3", "source": "c.md", "section": "s3",
         "content": "信用卡逾期会上报个人征信，影响房贷审批。"},
    ])
    return idx


# ── 分词 ────────────────────────────────────────

def test_continuous_chinese_is_segmented():
    """连续中文必须切成多段, 不能整体成一个 token"""
    toks = BM25Index()._tokenize("存款保险")
    assert len(toks) > 1, f"连续中文应切分, 实得 {toks}"


def test_bigram_split_shape():
    assert BM25Index()._tokenize("存款保险") == ["存款", "款保", "保险"]


def test_ascii_kept_as_whole_word():
    assert BM25Index()._tokenize("R2") == ["r2"]
    assert BM25Index()._tokenize("QDII基金") == ["qdii", "基金"]


def test_punctuation_splits_runs():
    toks = BM25Index()._tokenize("理财，非存款")
    assert "理财" in toks and "非存" in toks


# ── 检索 ────────────────────────────────────────

def test_natural_question_without_spaces_finds_docs():
    """核心回归: 没有空格的自然问句必须能检索到 (原实现返回 0 条)"""
    idx = _index()
    hits = idx.search("存款保险怎么赔", top_k=5)
    assert hits, "无空格中文问句返回 0 条 —— 分词 bug 复发了"
    assert any(h["source"] == "a.md" for h in hits)


def test_search_returns_real_source_filename():
    """source 必须是文件名, 不能是 chunk_id"""
    for h in _index().search("信用卡逾期影响房贷吗", top_k=5):
        assert h["source"].endswith(".md"), f"source 应为文件名, 实得 {h['source']!r}"
        assert h["source"] != h["chunk_id"], "source 不应等于 chunk_id"


def test_source_and_chunk_id_both_present():
    hits = _index().search("存款保险", top_k=5)
    pair = {(h["chunk_id"], h["source"]) for h in hits}
    assert pair, "应至少命中一条"
    assert pair <= {("c1", "a.md"), ("c2", "b.md")}, f"命中了不相关文档: {pair}"
    assert all(not cid.startswith("a.") for cid, _ in pair), "chunk_id/source 不能错位"


# ── 持久化往返 ──────────────────────────────────

def test_save_load_roundtrip_keeps_sources(tmp_path):
    idx = _index()
    p = str(tmp_path / "bm25_index.pkl")
    idx.save(p)

    loaded = BM25Index()
    loaded.load(p)
    assert loaded.doc_sources == ["a.md", "b.md", "c.md"]
    assert loaded.doc_sections == ["s1", "s2", "s3"]
    assert {h["source"] for h in loaded.search("存款保险", top_k=5)} == {"a.md", "b.md"}


def test_search_returns_section():
    """section 要和 FAISS / Milvus 两侧对齐, 否则前端来源标注缺一截"""
    h = _index().search("存款保险", top_k=1)[0]
    assert h["section"] in {"s1", "s2"}
