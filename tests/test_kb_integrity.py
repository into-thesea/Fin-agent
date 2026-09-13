"""
知识库链路完整性回归测试

固化 2026-09-13「先修坏的」这一轮修掉的不变量:

  1. 降级不许静默 —— 检索后端不可用/稀疏路为空时必须报错, 不能返回空结果
     让上层误以为「没找到」。尤其 hybrid_retrieve 不能退回纯向量(那正是
     「假混合检索」缺陷的成因)。
  2. 分块文件是追加型 —— 删文档必须同步清理它, 否则下次任何重建会让已删文档复活。
  3. 上传去重必须在落盘之前 —— 否则同名同内容重复上传会删掉已入库文档的源文件。
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import numpy as np
import pytest


# ──────────────────────────────────────────────
# 1. 降级不许静默
# ──────────────────────────────────────────────

class TestNoSilentDegradation:

    def test_milvus_search_raises_instead_of_returning_empty(self, monkeypatch):
        """Milvus 不可用时必须抛错 —— 返回 [] 会让用户看到「未找到资料」
        而不是「检索服务故障」, 前者是错误答案, 后者是可修复的故障。"""
        from src.vectorstore.milvus_manager import MilvusManager, MilvusUnavailableError

        mgr = MilvusManager()
        monkeypatch.setattr(type(mgr), "available", property(lambda self: False))

        with pytest.raises(MilvusUnavailableError):
            mgr.search(np.zeros((1, 768), dtype="float32"), top_k=3)
        with pytest.raises(MilvusUnavailableError):
            mgr.add_embeddings(np.zeros((1, 768), dtype="float32"), [{"content": "x"}])
        with pytest.raises(MilvusUnavailableError):
            mgr.delete_by_document("x.pdf")

    def test_bm25_search_raises_when_index_missing(self):
        """BM25 索引未加载时不能再 `except → return []`。"""
        from src.retrieval.retriever import HybridRetriever

        r = HybridRetriever()
        saved, r.bm25 = r.bm25, None
        try:
            with pytest.raises(RuntimeError, match="BM25 索引未加载"):
                r.bm25_search("测试查询")
        finally:
            r.bm25 = saved

    def test_hybrid_retrieve_raises_when_sparse_returns_empty(self):
        """稀疏路返回 0 条时必须报错, 不能静默退回纯向量检索。

        这是「假混合检索」缺陷的执行点: 只要 BM25 挂掉而稠密路有条,
        系统就会假装自己是混合检索。
        """
        from src.retrieval.retriever import HybridRetriever

        r = HybridRetriever()
        saved_vec, saved_bm = r.vector_search, r.bm25_search
        r.vector_search = lambda *a, **k: [{"content": "命中但来自稠密路", "chunk_id": "c1"}]
        r.bm25_search = lambda *a, **k: []
        try:
            with pytest.raises(RuntimeError, match="BM25 稀疏路返回 0 条"):
                r.hybrid_retrieve("测试查询")
        finally:
            r.vector_search, r.bm25_search = saved_vec, saved_bm

    def test_graph_retriever_raises_without_neo4j(self, monkeypatch):
        """Neo4j 不可用时不再降级内存图 —— 两个后端的图可能不一致。"""
        from src.knowledge_graph.core import Neo4jGraph
        from src.knowledge_graph.retriever import GraphRetriever

        monkeypatch.setattr(Neo4jGraph, "available", property(lambda self: False))
        with pytest.raises(RuntimeError, match="Neo4j 不可用"):
            GraphRetriever()

    def test_graph_retriever_memory_backend_is_explicit(self):
        """内存图仍可显式指定 (单测/离线调试用), 但不是自动降级路径。"""
        from src.knowledge_graph.retriever import GraphRetriever

        r = GraphRetriever(backend="memory")
        assert r._graph is not None

    def test_no_faiss_backend_fallback(self):
        """向量后端不得再有 FAISS 降级分支 —— 两种后端会给出不同答案,
        且 FAISS 路径下上传的新文档根本进不去。

        只查代码 (import / 调用), 不查文案 —— 注释里说明「为什么不再降级 FAISS」
        是应该保留的。
        """
        import inspect
        from src.retrieval import retriever as retriever_mod

        module_src = inspect.getsource(retriever_mod)
        assert "faiss_manager import" not in module_src
        assert "FaissIndexManager(" not in module_src


# ──────────────────────────────────────────────
# 2. 删除必须清掉分块文件 (否则重建复活)
# ──────────────────────────────────────────────

class TestPurgeChunks:

    def test_remove_chunks_of_purges_only_target_document(self, tmp_path, monkeypatch):
        """删除必须清掉 chunks_processed.jsonl 里该文档的分块。

        这是原先完全缺失的一步: 文件是追加型, 不清它的话下一次任何重建
        (BM25 / 向量) 都会把已删文档的分块重新写回索引 —— 删除看起来生效了,
        重建一次全复活。
        """
        from src.api.routes import knowledge

        store = tmp_path / "chunks_processed.jsonl"
        rows = [
            {"chunk_id": "k1", "source": "keep.pdf", "content": "保留"},
            {"chunk_id": "d1", "source": "drop.pdf", "content": "待删1"},
            {"chunk_id": "d2", "source": "drop.pdf", "content": "待删2"},
        ]
        store.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                         encoding="utf-8")
        monkeypatch.setattr("src.infra.paths.CHUNKS_PROCESSED_PATH", str(store))

        assert knowledge._remove_chunks_of("drop.pdf") == 2

        remaining = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert [c["chunk_id"] for c in remaining] == ["k1"]
        # 重建只能看到剩下的 —— 已删文档不会复活
        assert {c["source"] for c in remaining} == {"keep.pdf"}
        # 幂等: 再删一次没有东西可删
        assert knowledge._remove_chunks_of("drop.pdf") == 0
        # 不存在的文档不影响文件
        assert knowledge._remove_chunks_of("nope.pdf") == 0

    def test_rebuild_keeps_uploaded_chunks(self, tmp_path, monkeypatch):
        """重建必须保留上传文档的分块 —— 只刷新内置知识库。

        collect_chunks 只认 data/finance_kb/*.md, 而 rebuild_local 覆盖写分块文件:
        改之前重建一次, 上传的 PDF 就从索引里没了, 而 documents 表原样不动 ——
        管理台显示正常(status=done, chunks_count=N), 实际搜不到。
        """
        import build_finance_kb as b
        from src.database import db_manager

        store = tmp_path / "chunks_processed.jsonl"
        old = [
            {"chunk_id": "kb_buy_1", "source": "buy_process.md", "content": "内置旧版"},
            {"chunk_id": "up_1", "source": "uploaded.pdf", "content": "上传件第 1 块"},
            # 同一块被追加过两次 (重复入库): 合并后只该留一份
            {"chunk_id": "up_2", "source": "uploaded.pdf", "content": "上传件第 2 块"},
            {"chunk_id": "up_2", "source": "uploaded.pdf", "content": "上传件第 2 块"},
            # 已删除的文档: documents 表里没有, 不许被重建复活
            {"chunk_id": "gone_1", "source": "deleted.pdf", "content": "已删文档"},
        ]
        store.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in old),
                         encoding="utf-8")
        monkeypatch.setattr("src.infra.paths.CHUNKS_PROCESSED_PATH", str(store))
        monkeypatch.setattr("src.infra.paths.BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
        monkeypatch.setattr(db_manager, "get_all_documents",
                            lambda: [{"filename": "uploaded.pdf"}])
        # BM25 重建与本用例无关, 换掉以免依赖分词器与真实索引文件
        from src.retrieval import bm25_index
        monkeypatch.setattr(bm25_index, "rebuild_bm25_index", lambda *a, **k: None)

        new = [{"chunk_id": "kb_buy_1", "source": "buy_process.md", "content": "内置改版"},
               {"chunk_id": "kb_buy_2", "source": "buy_process.md", "content": "内置新增"}]
        n = b.rebuild_local(new)

        rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert n == len(rows) == 4, f"返回值要与落盘条数一致, 实际 n={n} rows={len(rows)}"
        ids = [c["chunk_id"] for c in rows]
        assert len(ids) == len(set(ids)), f"同一块被写了多遍: {ids}"
        # 内置: 刷新成新版 (旧内容被覆盖, 不是追加)
        assert [c["content"] for c in rows if c["chunk_id"] == "kb_buy_1"] == ["内置改版"]
        # 上传: 原样保留
        assert {c["chunk_id"] for c in rows if c["source"] == "uploaded.pdf"} == {"up_1", "up_2"}
        # 已删文档: 不复活
        assert "gone_1" not in ids

    def test_rebuild_keeps_nothing_when_document_table_is_empty(self, tmp_path, monkeypatch):
        """documents 表里没有的文档, 重建时一律丢弃 —— 判据是表, 不是后缀。"""
        import build_finance_kb as b
        from src.database import db_manager

        store = tmp_path / "chunks_processed.jsonl"
        store.write_text(json.dumps({"chunk_id": "x", "source": "some.pdf"}, ensure_ascii=False) + "\n",
                         encoding="utf-8")
        monkeypatch.setattr("src.infra.paths.CHUNKS_PROCESSED_PATH", str(store))
        monkeypatch.setattr("src.infra.paths.BM25_INDEX_PATH", str(tmp_path / "bm25.pkl"))
        monkeypatch.setattr(db_manager, "get_all_documents", lambda: [])
        from src.retrieval import bm25_index
        monkeypatch.setattr(bm25_index, "rebuild_bm25_index", lambda *a, **k: None)

        n = b.rebuild_local([{"chunk_id": "kb_1", "source": "buy_process.md", "content": "内置"}])

        rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert n == 1 and [c["chunk_id"] for c in rows] == ["kb_1"]

    def test_delete_uses_filename_as_vector_key(self):
        """删除向量时必须用 filename —— 索引里 document_id 存的就是文件名。

        用 SQLite 自增 id 会永远匹配不上(实际删除 0 条, 删除看似成功)。
        """
        import inspect
        from src.api.routes import knowledge

        src = inspect.getsource(knowledge.delete_document)
        assert "delete_by_document(filename)" in src
        assert "delete_by_document(str(doc_id))" not in src


# ──────────────────────────────────────────────
# 3. 上传安全
# ──────────────────────────────────────────────

class TestUploadSafety:

    def test_process_upload_does_not_write_before_dedup(self):
        """去重通过前不能落盘。

        原先先写盘再去重、重复就 os.remove —— 同名同内容重复上传时, 盘上那个
        文件正是**已入库文档的 PDF 源文件**, 会被删掉。
        """
        import inspect
        from src.api.routes import knowledge

        src = inspect.getsource(knowledge._process_upload)
        dedup_at = src.index("find_document_by_hash")
        write_at = src.index('open(file_path, "wb")')
        assert dedup_at < write_at, "去重必须发生在落盘之前"

    def test_upload_requires_celery_worker(self):
        """Celery Worker 不在时必须拒绝上传, 不再降级到后台线程。"""
        import inspect
        from src.api.routes import knowledge

        src = inspect.getsource(knowledge._process_upload)
        assert "is_celery_worker_running" in src
        assert "threading.Thread" not in src, "同步线程降级路径必须已删除"

    def test_batch_upload_rejects_over_limit(self):
        """并发闸: 超出单批文件数上限必须明确拒绝, 而不是全量并行压进内存与队列。"""
        from fastapi.testclient import TestClient
        from src.api.main import app
        from src.api.routes.auth import create_token
        from src.api.routes.knowledge import MAX_BATCH_FILES

        with TestClient(app) as c:
            files = [
                ("files", (f"f{i}.pdf", b"%PDF-1.4 fake", "application/pdf"))
                for i in range(MAX_BATCH_FILES + 1)
            ]
            r = c.post(
                "/api/v1/knowledge/batch-upload",
                files=files,
                headers={"Authorization": f"Bearer {create_token('admin', role='admin')}"},
            )
        assert r.status_code == 400
        assert "最多上传" in r.json()["detail"]

    def test_duplicate_claim_happens_before_any_index_write(self):
        """DB 占位必须发生在写任何索引之前 —— 这是幂等的关键。

        原先 DB 记录在流水线末尾才写, 两条并发的流水线都能通过去重检查, 各自往
        jsonl 与 Milvus 写一份向量, 最后只留一行记录 —— 索引里多出来的那些向量
        再也没东西能对应上, 删除时清不掉。
        """
        import inspect
        from src.core import pipeline_manager

        src = inspect.getsource(pipeline_manager.ingest_document)
        claim_at = src.index("占位登记")
        append_at = src.index("_append_chunks_to_store(chunks)")
        vector_at = src.index("add_chunks_to_index(chunks")
        assert claim_at < append_at, "占位必须早于分块写入"
        assert claim_at < vector_at, "占位必须早于向量写入"


class TestTaskStatusNormalization:

    def test_celery_states_map_to_canonical_vocabulary(self):
        """Celery 的大写状态必须映射到全系统统一的四个取值。

        它直接 lower() 出来是 success / progress, 不在前端认的集合里, 页面会一直
        卡在「进行中」—— 比直接显示失败更难排查。
        """
        from src.api.routes.tasks import _normalize_state, TASK_STATUS

        assert set(TASK_STATUS) == {"pending", "processing", "completed", "failed"}
        assert _normalize_state("SUCCESS") == "completed"
        assert _normalize_state("FAILURE") == "failed"
        assert _normalize_state("REVOKED") == "failed"
        assert _normalize_state("PROGRESS") == "processing"
        assert _normalize_state("STARTED") == "processing"
        assert _normalize_state("PENDING") == "pending"

    def test_already_canonical_values_pass_through(self):
        from src.api.routes.tasks import _normalize_state

        for v in ("pending", "processing", "completed", "failed"):
            assert _normalize_state(v) == v

    def test_unknown_state_is_not_passed_through(self):
        """未知状态不能原样透传 —— 前端会一直显示「进行中」。"""
        import logging
        from src.api.routes.tasks import _normalize_state

        # _normalize_state 对未知值会打 ERROR 日志。项目的 JSON 日志配置在 pytest
        # 的捕获 handler 下格式化这条记录会抛异常(经查是测试环境问题, 非产品缺陷;
        # 单独运行本用例通过)。这里临时把这个 logger 抬到 CRITICAL 以上, 让记录
        # 根本不进入 handler, 从而只测函数本身的行为。
        lg = logging.getLogger("src.api.routes.tasks")
        old_level = lg.level
        lg.setLevel(logging.CRITICAL + 1)
        try:
            assert _normalize_state("WEIRD_NEW_STATE") == "failed"
            assert _normalize_state(None) == "failed"
            assert _normalize_state("") == "failed"
        finally:
            lg.setLevel(old_level)
