"""
P7: FastMCP 服务化 — 把金融理财产品业务工具封装为独立 MCP Server (stdio)

把 finance_services 的只读工具 + retrieve_knowledge 标准化为 MCP 工具，
任何 MCP client (Python SDK / Claude Desktop / 其他 Agent) 都能调用同一批能力。
与主 FastAPI 应用解耦：本文件作为独立子进程 (stdio 传输) 运行。

安全边界: 资金动账(申购/赎回/撤单)不暴露为 MCP 工具 —— 由主应用 cs_graph 的
确定性门控节点执行并留痕。MCP 仅提供只读能力 (产品/适当性/持仓/知识库)。

运行 (stdio):
  .venv/Scripts/python.exe src/mcp/server.py

验证:
  .venv/Scripts/python.exe scripts/test_mcp.py
"""

from __future__ import annotations

import os
import sys

# 直接运行/被 stdio 拉起时把项目根加入路径 (兼容 from src... 导入)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastmcp import FastMCP

# 顶部导入工具实现：重型原生库 (numpy/scipy/sklearn/torch/transformers) 必须在主线程
# 启动时初始化——fastmcp 在工作线程执行工具，惰性首次导入 C 扩展会死锁。
import torch  # noqa: F401
import faiss  # noqa: F401
from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: F401

from src.business.context import set_current_user
from src.business.finance_services import (
    query_products as _query_products,
    check_suitability as _check_suitability,
    query_holdings as _query_holdings,
)
from src.retrieval.retriever import HybridRetriever  # noqa: F401
from src.tools.registry import retrieve_knowledge as _retrieve_knowledge

mcp = FastMCP("kefu-finance")


def _bind(user_id: str) -> None:
    """MCP 子进程无请求线程上下文, 工具显式接收 user_id 并绑定 (演示账号默认 u001)"""
    set_current_user(user_id or "u001")


@mcp.tool()
def query_products(product_type: str = "", risk: str = "", keyword: str = "") -> str:
    """按产品类型/风险等级(R1~R5)/关键词查询在售理财产品清单 (结构化实时目录)"""
    return _query_products(product_type=product_type, risk=risk, keyword=keyword).get("text", "")


@mcp.tool()
def check_suitability(product_id_or_name: str, user_id: str = "u001") -> str:
    """校验某产品与指定用户(演示账号 u001/u002/u003)的风险测评等级是否匹配 (适当性确定性判定)"""
    _bind(user_id)
    r = _check_suitability(product_id_or_name)
    return r.get("text", "") + ("\n[handoff]" if r.get("handoff") else "")


@mcp.tool()
def query_holdings(user_id: str = "u001") -> str:
    """查询指定用户(演示账号 u001/u002/u003)的理财持仓 (只读本人数据)"""
    _bind(user_id)
    return _query_holdings().get("text", "")


@mcp.tool()
def retrieve_knowledge(query: str, top_k: int = 5) -> str:
    """检索理财知识库 (产品说明书/政策条款/收益口径), 返回原文片段与来源"""
    return _retrieve_knowledge(query, top_k).get("text", "")


if __name__ == "__main__":
    # stdio 传输 (显式指定, 关横幅避免干扰), 与主应用解耦
    mcp.run(transport="stdio", show_banner=False)
