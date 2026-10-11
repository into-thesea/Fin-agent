"""审核入库必须可重试 —— 重试是**覆盖**同一窗口, 不是再追加一遍

回归点: approve_document 文档里写的"要么全部成功, 要么全部不入库"只覆盖 status,
不覆盖分块文件。Step 1 写文件成功、Step 2 挂掉时, 数据已落盘且没有回滚, 重试再
追加一遍同一份内容就成了双份 —— 实测一次失败 + 一次重试之后, 同一份文档在
chunks_processed.jsonl 里占了 130 行而不是 65 行。

键用 pending_doc_id 而不是 source 文件名: 两份同名文档各写各的, 按文件名会互相误删
(删除路径 _remove_chunks_of 就是按文件名, 已经踩过这个坑)。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import pipeline_manager as pm


@pytest.fixture
def store(tmp_path, monkeypatch):
    p = tmp_path / "chunks.jsonl"
    monkeypatch.setattr("src.infra.paths.CHUNKS_PROCESSED_PATH", str(p))
    return p


def _rows(p):
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _chunk(cid, doc_id, source="x.pdf"):
    return {"chunk_id": cid, "content": f"c-{cid}", "source": source,
            "chunk_type": "child", "pending_doc_id": doc_id}


def test_retry_overwrites_instead_of_appending(store):
    chunks = [_chunk("a_c1", 7), _chunk("a_c2", 7)]
    pm._append_chunks_to_store(chunks, replace_doc_id=7)
    assert len(_rows(store)) == 2

    pm._append_chunks_to_store(chunks, replace_doc_id=7)   # 重试
    rows = _rows(store)
    assert len(rows) == 2, "重试又追加了一遍 —— 这就是那次 130 行的成因"
    assert [r["chunk_id"] for r in rows] == ["a_c1", "a_c2"]


def test_replace_does_not_touch_other_documents(store):
    """两份同名文档各写各的, 互不误伤"""
    pm._append_chunks_to_store([_chunk("n_c1", 7), _chunk("n_c1", 8)], replace_doc_id=7)
    pm._append_chunks_to_store([_chunk("n_c1", 7)], replace_doc_id=7)
    rows = _rows(store)
    assert sorted(r["pending_doc_id"] for r in rows) == [7, 8]
    assert len(rows) == 2, f"另一份文档的行被误删了: {rows}"


def test_without_replace_id_it_is_still_append(store):
    """不传 replace_doc_id 时行为不变 (ingest_document 那条路径没传)"""
    pm._append_chunks_to_store([_chunk("a_c1", None)])
    pm._append_chunks_to_store([_chunk("a_c1", None)])
    assert len(_rows(store)) == 2


def test_corrupt_lines_survive_replace(store):
    """损坏行保留 —— 与 read_lines 的既有行为一致, 不因为覆盖写就顺手丢掉"""
    pm._append_chunks_to_store([_chunk("a_c1", 7)], replace_doc_id=7)
    with open(store, "a", encoding="utf-8") as f:
        f.write("{坏行\n")
    pm._append_chunks_to_store([_chunk("a_c2", 7)], replace_doc_id=7)
    assert "{坏行" in store.read_text(encoding="utf-8")
    assert [r["chunk_id"] for r in _rows(store)] == ["a_c2"]
