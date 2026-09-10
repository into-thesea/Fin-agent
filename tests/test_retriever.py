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
