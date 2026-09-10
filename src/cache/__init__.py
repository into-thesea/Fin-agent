# Fin-Agent 缓存层
from src.cache.redis_client import RedisCache, semantic_cache_key

# 全局缓存单例（懒加载）
_cache_instance = None

def get_cache() -> "RedisCache":
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance

def reset_cache():
    """用于测试或手动刷新"""
    global _cache_instance
    if _cache_instance:
        _cache_instance.close()
        _cache_instance = None

__all__ = ["RedisCache", "get_cache", "reset_cache", "semantic_cache_key"]
