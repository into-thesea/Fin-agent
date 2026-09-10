"""
Fin-Agent 数据库层

包结构:
  - manager.py  — DBManager 单例 (连接管理、文档CRUD、审计、任务追踪)
"""

from src.db.manager import DBManager, db_manager

__all__ = ["DBManager", "db_manager"]
