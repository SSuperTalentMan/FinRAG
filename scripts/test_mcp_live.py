#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/test_mcp_live.py — MCP 工具层真实联调（需 FinRag 服务已在 :8000 运行）。

前置：
1. 服务启动（带 Agentic 开关）：
   SKIP_CONFIG_VALIDATION=1 ORCH_AGENTIC_MODE=1 .venv/Scripts/python.exe main.py
2. 签发 token（本机测试账号）：
   SKIP_CONFIG_VALIDATION=1 .venv/Scripts/python.exe -c "from routers.auth import _create_token; open('logs/_mcp_token.txt','w').write(_create_token(1,'admin_test','admin'))"

运行：.venv/Scripts/python.exe scripts/test_mcp_live.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

token = (Path("logs/_mcp_token.txt")).read_text().strip()
from rag_qa.mcp_layer.client import McpClient

client = McpClient(env={"FINRAG_TOKEN": token})
client.start()
tools = client.list_tools()
print(f"[握手] tools={len(tools)}: {[t['name'] for t in tools]}\n")


def show(name, args):
    r = client.call_tool(name, args, timeout=180)
    ok = r.get("ok")
    m = r.get("meta", {})
    print(f"[{name}] ok={ok} engine={m.get('engine')} elapsed={m.get('elapsed_ms')}ms")
    if ok:
        d = r.get("data", {})
        for k, v in list(d.items())[:4]:
            print(f"    {k}: {str(v)[:140]}")
    else:
        print(f"    error: {str(r.get('error'))[:180]}")
    print()
    return r


# 1) 知识库检索（真实 Milvus+BM25+精排，无需 LLM）
show("finrag.kb_search", {"question": "存款保险最高赔付多少", "top_k": 3})

# 2) 端到端问答（agentic_mode=1：决策需 LLM → 验证回退固定路由链路；生成无 Key → 验证降级）
r = show("finrag.ask", {"question": "什么是存款保险"})

# 3) 问数（SQL 生成需 LLM → 验证优雅失败）
show("finrag.sql_query", {"question": "上月华东大区销售额多少"})

# 4) 审查状态（不存在的任务 → 验证真实 404 透传）
show("finrag.review_status", {"task_id": 999999})

client.close()
print("DONE")
