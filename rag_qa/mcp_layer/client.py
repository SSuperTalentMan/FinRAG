#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/mcp_layer/client.py — MCP stdio 子进程客户端（纯标准库）。

用法：
    from rag_qa.mcp_layer.client import McpClient
    client = McpClient()          # 默认用当前解释器启动 finrag-mcp
    client.start()                # 握手 + notifications/initialized
    tools = client.list_tools()
    result = client.call_tool("finrag.ask", {"question": "存款保险最高赔多少"})

防御：读响应时跳过任何不以 { 开头的行（防库改道不彻底漏出的脏输出）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading


class McpError(RuntimeError):
    pass


class McpClient:
    def __init__(self, server_cmd: list[str] | None = None, env: dict | None = None):
        self._cmd = server_cmd or [sys.executable, "-m", "rag_qa.mcp_layer.server_stdio"]
        self._env = env or {}
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self._lock = threading.Lock()

    def start(self) -> dict:
        """启动子进程并完成握手，返回 initialize 结果。"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))  # 项目根
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.update(self._env)
        self._proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, text=True, encoding="utf-8",
        )
        result = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "finrag-bridge", "version": "1.0.0"},
        })
        self._notify("notifications/initialized", {})
        return result

    def list_tools(self) -> list[dict]:
        return self._request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict, timeout: float = 300.0) -> dict:
        """返回 {ok, data|error, meta}（tools.py 的 ToolResult.to_dict 结构）。"""
        resp = self._request("tools/call", {"name": name, "arguments": arguments},
                             timeout=timeout)
        content = resp.get("content") or []
        if not content:
            raise McpError(f"工具 {name} 返回空内容")
        return json.loads(content[0]["text"])

    # -- 内部 --
    def _send(self, payload: dict) -> None:
        assert self._proc and self._proc.stdin
        with self._lock:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _read_line(self, timeout: float) -> str:
        assert self._proc and self._proc.stdout
        # 简单超时：读线程 + deadline（Windows 下 select 不支持管道）
        import time

        deadline = time.monotonic() + timeout
        box: list[str] = []

        def reader():
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            box.append(line)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(max(0.1, deadline - time.monotonic()))
        if not box or not box[0]:
            raise McpError(f"等待服务响应超时（>{timeout}s）")
        return box[0]

    def _request(self, method: str, params: dict, timeout: float = 60.0) -> dict:
        self._next_id += 1
        req_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        while True:
            line = self._read_line(timeout).strip()
            if not line.startswith("{"):
                continue  # 跳过脏输出（日志改道不彻底的兜底防御）
            msg = json.loads(line)
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise McpError(f"{method} 失败: {msg['error'].get('message')}")
                return msg.get("result", {})

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
