"""
检索器单元测试 (不需要外部依赖的测试)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pytest


class TestHybridRetriever:
    """混合检索器测试"""

    def test_import(self):
        """验证导入 (无 sentence_transformers 时优雅降级)"""
        try:
            from src.retrieval.retriever import HybridRetriever
            assert HybridRetriever is not None
        except ModuleNotFoundError as e:
            # sentence_transformers 不可用 — 跳过 (CI/Docker 环境会有)
            pytest.skip(f"依赖缺失: {e}")

    def test_faiss_manager_singleton(self):
        """验证 FaissIndexManager 单例"""
        from src.cache.faiss_manager import FaissIndexManager
        m1 = FaissIndexManager()
        m2 = FaissIndexManager()
        assert m1 is m2  # 同一实例

    def test_faiss_manager_noop(self):
        """验证 FAISS 检索安全 (bge-base-zh-v1.5 = 768 维; 索引存在返回列表, 否则空)"""
        from src.cache.faiss_manager import FaissIndexManager
        mgr = FaissIndexManager()
        results = mgr.search(np.array([[0.1] * 768], dtype="float32"), top_k=3)
        assert isinstance(results, list)

    def test_faiss_manager_reset(self):
        """验证重置"""
        from src.cache.faiss_manager import FaissIndexManager
        mgr = FaissIndexManager()
        mgr.reset()
        assert mgr.total_count == 0

    def test_faiss_manager_reload(self):
        """验证重新加载 (无文件时安全)"""
        from src.cache.faiss_manager import FaissIndexManager
        mgr = FaissIndexManager()
        mgr.reload()  # 不抛出异常即可
        assert True


class TestRetrievalConfigWiring:
    """策略页上可编辑的每个检索参数都必须真的驱动行为"""

    def test_rrf_k_comes_from_config(self, monkeypatch):
        from src.retrieval import retriever as mod
        from src.core import kb_settings

        seen = {}
        monkeypatch.setattr(mod, "rrf_fusion",
                            lambda d, s, top_k=5, k=60: seen.update(k=k) or [])
        monkeypatch.setattr(mod.HybridRetriever, "vector_search",
                            lambda self, *a, **kw: [{"chunk_id": "a", "content": "x"}])
        monkeypatch.setattr(mod.HybridRetriever, "bm25_search",
                            lambda self, *a, **kw: [{"chunk_id": "a", "content": "x"}])
        monkeypatch.setattr(mod.HybridRetriever, "_graph_retrieve",
                            lambda self, q: {"entries": [], "entities": []})
        # retriever 内部是"函数内 import", 所以 patch 模块属性即可生效
        monkeypatch.setattr(kb_settings, "get_retrieval",
                            lambda: {**kb_settings.DEFAULTS["retrieval"], "rrf_k": 17})

        mod.HybridRetriever().hybrid_retrieve("测试", top_k=5)
        assert seen.get("k") == 17, "rrf_k 没有从配置传进 rrf_fusion"

    def test_mmr_lambda_comes_from_config(self, monkeypatch):
        """MMR 的 λ 目前硬编码 0.5, 必须改成读配置"""
        from src.retrieval import retriever as mod
        import inspect as _i
        src = _i.getsource(mod.HybridRetriever.vector_search)
        assert "lambd=0.5" not in src, "λ 仍是硬编码, 配置改了不生效"
        assert "lambd=" in src and ("cfg" in src or "get_retrieval" in src)

    def test_top_k_comes_from_config_but_explicit_arg_wins(self, monkeypatch):
        """不传 top_k 的调用方跟随配置; 显式传的调用方行为不变"""
        from src.retrieval import retriever as mod
        from src.core import kb_settings

        seen = {}
        monkeypatch.setattr(mod, "rrf_fusion",
                            lambda d, s, top_k=5, k=60: seen.update(top_k=top_k) or [])
        monkeypatch.setattr(mod.HybridRetriever, "vector_search",
                            lambda self, *a, **kw: [{"chunk_id": "a", "content": "x"}])
        monkeypatch.setattr(mod.HybridRetriever, "bm25_search",
                            lambda self, *a, **kw: [{"chunk_id": "a", "content": "x"}])
        monkeypatch.setattr(mod.HybridRetriever, "_graph_retrieve",
                            lambda self, q: {"entries": [], "entities": []})
        monkeypatch.setattr(kb_settings, "get_retrieval",
                            lambda: {**kb_settings.DEFAULTS["retrieval"], "top_k": 7})

        r = mod.HybridRetriever()
        r.hybrid_retrieve("测试")                    # 不传 → 跟随配置
        assert seen.get("top_k") == 7, "top_k 没有从配置生效"

        r.hybrid_retrieve("测试", top_k=3)           # 显式传 → 不变
        assert seen.get("top_k") == 3, "显式 top_k 被配置覆盖了"
