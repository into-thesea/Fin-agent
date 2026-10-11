"""父块不进检索索引 —— 三条索引路径必须同口径

回归点: 全量向量重建和 BM25 重建都硬编码过滤了父块, 而增量 add_chunks_to_index
漏了。于是每审核通过一份上传文档, 父块就被灌进 Milvus, 整章大块在 top-k 里跟
子块抢位, 直到下次全量重建才被清掉 —— 而全量重建会 mgr.reset(), 所以这个不一致
平时看不出来, 只在"入库之后、下次重建之前"这段窗口里生效。

实测依据: 417 个父块里 228 个(55%)与某个子块逐字相同; 最长父块 8744 字,
远超 bge-base-zh-v1.5 的 512 token 上下文。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.smart_chunk import indexable_chunks
from src.retrieval import vector_indexer as vi


def _chunk(cid: str, kind: str) -> dict:
    return {"chunk_id": cid, "content": f"内容-{cid}", "source": "x.md", "chunk_type": kind}


def test_indexable_chunks_drops_parents():
    chunks = [_chunk("p1", "parent"), _chunk("p1_c1", "child"), _chunk("p1_c2", "child")]
    assert [c["chunk_id"] for c in indexable_chunks(chunks)] == ["p1_c1", "p1_c2"]


def test_indexable_chunks_keeps_chunks_without_type():
    """旧格式分块(没有 chunk_type)必须原样保留 —— 判据是"等于 parent", 不是"不等于 child" """
    legacy = [{"chunk_id": "old1", "content": "x"}, {"chunk_id": "old2", "content": "y"}]
    assert indexable_chunks(legacy) == legacy


def test_incremental_index_skips_parents(monkeypatch):
    """审核入库走的就是这条路 (approve_document → add_chunks_to_index)"""
    captured = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured["texts"] = list(texts)
            return np.zeros((len(texts), 4), dtype="float32")

    class FakeMilvus:
        def delete_by_chunk_ids(self, chunk_ids):
            captured["deleted"] = list(chunk_ids)
            return 0

        def add_embeddings(self, embeddings, metadata):
            captured["meta"] = list(metadata)

        @property
        def total_count(self):
            return len(captured.get("meta") or [])

    monkeypatch.setattr(vi, "_get_model", lambda: FakeModel())
    monkeypatch.setattr("src.vectorstore.milvus_manager.MilvusManager", FakeMilvus)

    mixed = [_chunk("p1", "parent"), _chunk("p1_c1", "child"),
             _chunk("p2", "parent"), _chunk("p2_c1", "child")]
    n = vi.add_chunks_to_index(mixed, document_id="x.md")

    assert captured["texts"] == ["内容-p1_c1", "内容-p2_c1"], "父块被编码进向量了"
    assert [m["chunk_id"] for m in captured["meta"]] == ["p1_c1", "p2_c1"]
    assert n == 2, "返回值是实际写入的条数, 不该把父块算进去"


def test_incremental_index_only_parents_writes_nothing(monkeypatch):
    """全是父块时应当直接返回, 连模型都不加载"""
    def _boom():
        raise AssertionError("全是父块时不该加载模型")

    monkeypatch.setattr(vi, "_get_model", _boom)
    assert vi.add_chunks_to_index([_chunk("p1", "parent")], document_id="x.md") == 0
