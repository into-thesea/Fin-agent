"""
缓存层单元测试

测试 RedisCache 的优雅降级、语义缓存、向量缓存等核心功能。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

import pytest


class TestRedisCache:
    """Redis 缓存层测试 (无 Redis 时验证优雅降级)"""

    def setup_method(self):
        import unittest.mock as mock
        from src.cache.redis_client import RedisCache
        # 模拟 Redis 不可用, 使降级测试不依赖真实环境 (Redis 现常驻运行)
        self._patch = mock.patch.object(RedisCache, "_check_available", return_value=False)
        self._patch.start()
        self.cache = RedisCache()

    def teardown_method(self):
        self._patch.stop()

    def test_graceful_degradation(self):
        """验证 Redis 不可用时优雅降级"""
        assert self.cache.enabled is False
        assert self.cache.ping() is False
        stats = self.cache.get_stats()
        assert stats["enabled"] is False

    def test_vector_cache_noop(self):
        """验证向量缓存降级行为"""
        result = self.cache.get_vector_results("test_hash", 5)
        assert result is None  # 降级时返回 None

        # 写入不应报错
        self.cache.set_vector_results("test_hash", 5, [{"content": "test"}])
        # 读取仍返回 None
        assert self.cache.get_vector_results("test_hash", 5) is None

    def test_graph_cache_noop(self):
        """验证图谱缓存降级行为"""
        result = self.cache.get_graph_relations("test_entity")
        assert result is None

        self.cache.set_graph_relations("test_entity", ["relation1"])
        assert self.cache.get_graph_relations("test_entity") is None

    def test_semantic_cache_noop(self):
        """验证语义缓存降级行为"""
        result = self.cache.get_semantic_response("测试查询")
        assert result is None

        self.cache.set_semantic_response("测试查询", "测试回答")
        assert self.cache.get_semantic_response("测试查询") is None

    def test_task_status_noop(self):
        """验证任务状态缓存降级行为"""
        self.cache.set_task_status("task_1", "processing", 50, "testing")
        result = self.cache.get_task_status("task_1")
        assert result is None

    def test_invalidate_noop(self):
        """验证缓存失效降级行为"""
        # 不应抛出异常
        self.cache.invalidate_entity_cache("test_entity")
        self.cache.invalidate_document_cache("test_doc")
        self.cache.invalidate_all()

    def test_semantic_cache_key(self):
        """验证语义缓存 key 生成"""
        from src.cache.redis_client import semantic_cache_key
        key1 = semantic_cache_key("比亚迪2024年营收")
        key2 = semantic_cache_key("比亚迪2024年营收")
        key3 = semantic_cache_key("特斯拉2024年营收")

        assert key1 == key2  # 相同输入 → 相同 key
        assert key1 != key3  # 不同输入 → 不同 key
        assert len(key1) == 32  # MD5 长度
        assert isinstance(key1, str)


class TestSimilarity:
    """余弦相似度测试"""

    def test_identical_texts(self):
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        sim = cache._cosine_sim_float("比亚迪2024年营收", "比亚迪2024年营收")
        assert sim == 1.0

    def test_similar_texts(self):
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        sim = cache._cosine_sim_float("比亚迪2024年营收", "BYD 2024 revenue")
        assert sim > 0  # 应有正相似度

    def test_different_texts(self):
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        sim = cache._cosine_sim_float("比亚迪营收", "特斯拉毛利率")
        assert sim >= 0

    def test_empty_strings(self):
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        assert cache._cosine_sim_float("", "") == 0.0
        assert cache._cosine_sim_float("test", "") == 0.0
        assert cache._cosine_sim_float("", "test") == 0.0
