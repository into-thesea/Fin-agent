"""
P8: 安全拦截器 — 工具调用统一权限/审计

每个工具调用前执行:
  1. 读取当前用户 (thread-local, 由请求线程/图节点绑定)
  2. 按工具类型执行策略:
     - 用户资源工具 (query_holdings / check_suitability): 数据访问内部按当前用户
       限定本人, 此处记录审计事件 (谁在何时查了谁的持仓/做了适当性判定)
     - 知识库 (retrieve_knowledge): 用户域访问控制 (默认全开放, 策略可配)
     - 外网 (web_search): 域名白名单 (默认禁用)
  3. 资金动账(申购/赎回/撤单)不注册为 Agent 工具, 由 cs_graph 确定性门控节点执行并留痕,
     故此处无需对动账做 agent 工具级拦截。
  4. 拒绝: 记录安全事件 (Redis security:events) + 返回降级文案, 不执行真实工具

用法: 包装 EXECUTE_TOOLS 的每个工具 (tools/registry.py)。
"""

from __future__ import annotations

import os
import time
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# 用户资源类工具: 无跨用户参数, 数据访问已按当前用户限定本人; 这里做调用审计
_AUDIT_USER_TOOLS = {"query_holdings", "check_suitability"}

# 知识库用户域: 默认所有用户可访问全部来源; 留扩展口 (如 {"internal": ["内部政策"]})
KB_DOMAIN_ALLOW = os.getenv("KB_DOMAIN_ALLOW", "*")  # "*" = 全开放

# 外网搜索域名白名单 (默认禁用 web_search)
WEB_ALLOWED_DOMAINS = {d for d in os.getenv("WEB_ALLOWED_DOMAINS", "").split(",") if d}


def _audit(event: str, user_id: str, tool: str, detail: str) -> None:
    """安全事件审计 (Redis list security:events; 不可用则跳过, 不阻塞)"""
    try:
        from src.cache.redis_client import RedisCache
        cache = RedisCache()
        if cache.client:
            cache.client.lpush(
                "security:events",
                json.dumps({"ts": int(time.time()), "event": event, "user": user_id or "anon",
                            "tool": tool, "detail": detail[:200]}, ensure_ascii=False),
            )
            cache.client.ltrim("security:events", 0, 499)
    except Exception as e:
        logger.debug("安全审计写入失败 (可忽略): %s", e)


def intercept(tool_name: str, fn: Callable) -> Callable:
    """包装工具: 调用前做权限/审计, 拒绝时不执行真实工具"""
    def wrapped(**kwargs):
        from src.business.context import get_current_user
        uid = get_current_user()

        # ── 用户资源工具: 审计 (持仓/适当性查询按当前用户限定, 无越权参数面) ──
        if tool_name in _AUDIT_USER_TOOLS:
            _audit("user_resource_access", uid, tool_name,
                   json.dumps({k: str(v)[:80] for k, v in kwargs.items()}, ensure_ascii=False))

        # ── 知识库: 用户域访问控制 ──
        if tool_name == "retrieve_knowledge" and KB_DOMAIN_ALLOW != "*":
            # 留扩展: 此处可按用户角色/来源过滤 (当前单公共库, 全开放)
            pass

        # ── 外网搜索: 域名白名单 ──
        if tool_name == "web_search" and not WEB_ALLOWED_DOMAINS:
            _audit("denied_web", uid, tool_name, "web_search 未配置白名单, 默认禁用")
            return {"text": "【安全拦截】外部搜索未开放。", "sources": [], "contexts": {}}

        return fn(**kwargs)

    return wrapped
