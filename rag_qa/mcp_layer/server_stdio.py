#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/mcp_layer/server_stdio.py — 最小 MCP stdio Server（纯标准库，零新增依赖）。

MCP stdio 传输 = 换行分隔的 JSON-RPC 2.0，只实现三个方法：
  initialize   → {protocolVersion, capabilities:{tools:{}}, serverInfo}
  tools/list   → {tools:[{name, description, inputSchema}]}
  tools/call   → {content:[{type:"text", text}], isError}

约定：id 为 None 的消息是通知（notification），不响应；未知方法回 -32601；
解析失败回 -32700。

⚠️ 最大的坑：stdout 被 MCP 协议独占 —— 一行非 JSON 日志就会让客户端解析失败。
必须在导入任何业务模块之前抢占 stdout（协议通道），并把 sys.stdout 指向 stderr，
让后续所有库的打印/日志自动改道。
"""
from __future__ import annotations

import json
import os
import sys

# ── stdout 改道：必须先于任何业务模块导入 ─────────────────────────────────────
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

from rag_qa.mcp_layer.registry import ToolResult  # noqa: E402
from rag_qa.mcp_layer.tools import build_registry  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "finrag-mcp", "version": "1.0.0"}

_registry = build_registry()
_caller = os.environ.get("FINRAG_CALLER", "agent")


def _resp(msg_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result},
                      ensure_ascii=False)


def _resp_err(msg_id, code, message):
    return json.dumps({"jsonrpc": "2.0", "id": msg_id,
                       "error": {"code": code, "message": message}},
                      ensure_ascii=False)


def _handle(msg: dict) -> str | None:
    """处理一条 JSON-RPC 消息；通知返回 None（不响应）。"""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if msg_id is None:
        return None  # notification：initialized / cancelled 等，一律不响应

    if method == "initialize":
        return _resp(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "ping":
        return _resp(msg_id, {})
    if method == "tools/list":
        return _resp(msg_id, {"tools": _registry.list_tools()})
    if method == "tools/call":
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        result: ToolResult = _registry.call(name, args, caller=_caller)
        text = json.dumps(result.to_dict(), ensure_ascii=False)
        return _resp(msg_id, {
            "content": [{"type": "text", "text": text}],
            "isError": not result.ok,
        })
    return _resp_err(msg_id, -32601, f"Method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _PROTOCOL_OUT.write(_resp_err(None, -32700, "Parse error") + "\n")
            _PROTOCOL_OUT.flush()
            continue
        out = _handle(msg)
        if out is not None:
            _PROTOCOL_OUT.write(out + "\n")
            _PROTOCOL_OUT.flush()


if __name__ == "__main__":
    main()
