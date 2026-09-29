#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/mcp_demo.py — FinRag MCP 工具层端到端演示。

三部分（A/B 离线可跑，不需要 FinRag 服务；C 的工具调用需要服务在跑）：
  A) 管控层负例：未知工具 / 缺参 / 类型错 / 身份越权 / 超时隔离 —— 证明管控真的在拦；
  B) 协议往返：真起子进程，握手 + tools/list + tools/call 错误处理 —— 证明 MCP 通道可用；
  C) 真实工具调用：finrag.kb_search / finrag.ask（服务未启动时给出明确降级说明）。

运行：.venv/Scripts/python.exe scripts/mcp_demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.mcp_layer.registry import Tool, ToolRegistry, ToolResult, mask_args


def banner(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def main() -> None:
    # ── A) 管控层负例 ────────────────────────────────────────────────────────
    banner("A) 管控层负例（离线）")
    reg = ToolRegistry(audit_path=os.path.join("logs", "mcp_demo_audit.jsonl"))
    reg.register(Tool(
        name="demo.echo",
        description="演示用回声工具",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                      "required": ["text"]},
        handler=lambda a: ToolResult(True, data={"echo": a["text"]}),
        timeout=2.0,
    ))

    def _slow(_: dict) -> ToolResult:
        time.sleep(5)
        return ToolResult(True, data="never")

    reg.register(Tool(
        name="demo.slow", description="演示超时隔离", input_schema={"type": "object", "properties": {}},
        handler=_slow, timeout=1.0,
    ))

    cases = [
        ("未知工具", lambda: reg.call("no.such", {}, caller="agent")),
        ("缺必填参数", lambda: reg.call("demo.echo", {}, caller="agent")),
        ("类型错误(bool 冒充 integer)",
         lambda: reg.call("demo.echo", {"text": 1}, caller="agent")),
        ("身份越权", lambda: reg.call("demo.echo", {"text": "hi"}, caller="anonymous")),
        ("超时隔离", lambda: reg.call("demo.slow", {}, caller="agent")),
    ]
    for name, fn in cases:
        r = fn()
        print(f"  [{name}] ok={r.ok} -> {r.error or r.data}")

    print("\n  审计脱敏示例：")
    print("   ", mask_args({"question": "存款保险最高赔多少", "api_key": "sk-abcdefgh12345678",
                          "phone": "13812345678"}))

    # ── B) 协议往返（子进程真握手）────────────────────────────────────────────
    banner("B) MCP 协议往返（子进程）")
    from rag_qa.mcp_layer.client import McpClient

    client = McpClient(env={"FINRAG_MCP_AUDIT": os.path.join("logs", "mcp_demo_audit.jsonl")})
    try:
        init = client.start()
        print(f"  握手成功: server={init['serverInfo']['name']} "
              f"protocol={init['protocolVersion']}")
        tools = client.list_tools()
        print(f"  tools/list: {len(tools)} 个工具")
        for t in tools:
            print(f"    - {t['name']}")
        r = client.call_tool("no.such.tool", {})
        print(f"  未知工具调用: ok={r['ok']} error={r.get('error', '')[:50]}")
    except Exception as e:  # noqa: BLE001
        print(f"  协议往返失败: {type(e).__name__}: {e}")
    finally:
        client.close()

    # ── C) 真实工具调用（需要 FinRag 服务在跑）────────────────────────────────
    banner("C) 真实工具调用（需服务已启动，未启动则验证降级路径）")
    if not os.environ.get("FINRAG_TOKEN"):
        print("  [提示] 未设置 FINRAG_TOKEN，调用会收到 401 —— 这也是正确的降级行为")
    try:
        client2 = McpClient(env={"FINRAG_MCP_AUDIT": os.path.join("logs", "mcp_demo_audit.jsonl")})
        client2.start()
        r = client2.call_tool("finrag.kb_search", {"question": "存款保险最高赔付多少", "top_k": 3})
        if r.get("ok"):
            d = r["data"]
            print(f"  finrag.kb_search: 命中 {d['hit_count']} 条，meta={r['meta']}")
            for s in d["sources"][:2]:
                print(f"    - [{s.get('score', 0):.3f}] {str(s.get('question', ''))[:40]}")
        else:
            print(f"  finrag.kb_search 降级: {r.get('error', '')[:120]}")
        client2.close()
    except Exception as e:  # noqa: BLE001
        print(f"  真实调用失败: {type(e).__name__}: {str(e)[:120]}")

    banner("演示结束")
    print("  开启 Agentic RAG：config.ini [orchestrator] agentic_mode = true（或环境变量 ORCH_AGENTIC_MODE=1）")


if __name__ == "__main__":
    main()
