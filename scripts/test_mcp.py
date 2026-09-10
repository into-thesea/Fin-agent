"""
P7: MCP client 演示 — 启动 FastMCP server (stdio), 列出工具并调用

用同步 subprocess + 裸 MCP JSON-RPC 协议 (Windows 下 asyncio subprocess stdio
存在兼容问题, 同步 readline 稳定)。顺带展示 MCP 报文, 便于理解协议。

用法:
  .venv/Scripts/python.exe scripts/test_mcp.py
"""

from __future__ import annotations

import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = subprocess.Popen(
        [sys.executable, "src/mcp/server.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, cwd=PROJECT_ROOT,
    )

    def send(msg: dict):
        p.stdin.write((json.dumps(msg) + "\n").encode())
        p.stdin.flush()

    def recv() -> dict:
        return json.loads(p.stdout.readline())

    try:
        send({"jsonrpc": "2.0", "id": 0, "method": "initialize",
              "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                         "clientInfo": {"name": "test", "version": "1"}}})
        init = recv()
        print("✅ initialize OK, protocol:", init["result"]["protocolVersion"])

        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        tools = recv()
        print("✅ MCP 工具列表:", [t["name"] for t in tools["result"]["tools"]])

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "query_products", "arguments": {"risk": "R2"}}})
        res = recv()
        text = res["result"]["content"][0]["text"]
        print("✅ call query_products(risk=R2):")
        print("   ", text[:160].replace(chr(10), " "))

        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "retrieve_knowledge", "arguments": {"query": "银行理财是存款吗"}}})
        res2 = recv()
        text2 = res2["result"]["content"][0]["text"]
        print("✅ call retrieve_knowledge('银行理财是存款吗'):")
        print("   ", text2[:160].replace(chr(10), " "))
    finally:
        p.terminate()


if __name__ == "__main__":
    main()
