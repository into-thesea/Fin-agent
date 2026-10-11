"""上传文档 → 三元组审核队列 的接线自检

回归点: `extract_triples.py` 早就写好了, 但全仓没有任何调用方 —— 新文档入库后
正文里的关系永远不会进图谱, 只能人肉补。这里守的是接线之后的三条语义:
只抽父块 / 剔掉图谱里已有的边 / 单文档限量。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from scripts import extract_triples as et


def _triple(head, rel, tail, doc):
    return {"head": head, "relation": rel, "tail": tail, "doc": doc,
            "source_text": "", "confidence": 0.0, "status": "pending"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """把待审目录 / 图谱快照挪到 tmp, LLM 换成固定返回"""
    pending = tmp_path / "kg_pending"
    kg = tmp_path / "kg_triples.jsonl"
    kg.write_text(json.dumps({"head": "理财产品", "relation": "has_fee", "tail": "管理费"},
                             ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(et, "PENDING_DIR", pending)
    monkeypatch.setattr(et, "KG_TRIPLES", kg)

    calls = []

    def fake_extract(text, doc_name):
        calls.append(text)
        return [
            _triple("理财产品", "has_fee", "管理费", doc_name),      # 图谱已有 → 该被剔掉
            _triple("结构性存款", "covered_by", "存款保险", doc_name),
            _triple("结构性存款", "covered_by", "存款保险", doc_name),  # 文档内重复 → 只留一条
        ]

    monkeypatch.setattr(et, "extract_from_text", fake_extract)
    return pending, calls


def _rows(pending, name):
    path = pending / name
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_only_parent_chunks_are_extracted(env):
    _, calls = env
    chunks = [
        {"chunk_type": "parent", "content": "父块A"},
        {"chunk_type": "child", "content": "子块A1"},
        {"chunk_type": "parent", "content": "父块B"},
        {"chunk_type": "child", "content": "子块B1"},
    ]
    stats = et.extract_chunks_to_pending(chunks, "01_江苏银行.pdf")
    assert calls == ["父块A", "父块B"], "子块与父块内容重叠, 抽了既多花调用又出重复三元组"
    assert stats["chunks"] == 2


def test_known_and_duplicate_triples_are_filtered(env):
    pending, _ = env
    stats = et.extract_chunks_to_pending([{"chunk_type": "parent", "content": "x"}],
                                         "01_江苏银行.pdf")
    assert stats["extracted"] == 3
    assert stats["skipped_known"] == 1, "图谱里已有的边不该再塞给审核员"
    assert stats["queued"] == 1
    rows = _rows(pending, "01_江苏银行.jsonl")
    assert len(rows) == 1
    assert rows[0]["tail"] == "存款保险"
    assert rows[0]["status"] == "pending", "写出来的必须是待审状态"


def test_max_chunks_caps_llm_calls(env):
    _, calls = env
    chunks = [{"chunk_type": "parent", "content": f"块{i}"} for i in range(10)]
    et.extract_chunks_to_pending(chunks, "big.pdf", max_chunks=3)
    assert len(calls) == 3, "worker 是 solo 串行, 不限量会把上传/预览一起堵住"


def test_utf8_guard_survives_celery_logging_proxy(monkeypatch):
    """Celery worker 里 sys.stdout 是 LoggingProxy, 没有 reconfigure

    实测踩过: 裸调用 reconfigure 让 import 这一步就炸, 任务连函数体都进不去,
    worker 还跟着挂了 —— 而进程内 apply() 测不出来 (普通进程的 stdout 是真对象)。
    """
    class NoReconfigure:                      # LoggingProxy 的最小复刻
        def write(self, s):
            pass

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", NoReconfigure())
    monkeypatch.setattr(sys, "stderr", NoReconfigure())
    et.force_utf8_console()                   # 不该抛


def test_doc_without_parent_chunks_falls_back_to_whole_text(env):
    """旧格式(没有父子分块)的文档不能抽不到东西"""
    _, calls = env
    chunks = [{"content": "A" * 900}, {"content": "B" * 900}]
    et.extract_chunks_to_pending(chunks, "legacy.txt")
    assert calls, "没有父块时应退化成整篇按 800 字切"
    assert all(len(c) <= 800 for c in calls)
