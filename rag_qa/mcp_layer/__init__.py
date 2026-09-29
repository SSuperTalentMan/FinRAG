#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/mcp_layer — 把 FinRag 能力封装为标准 MCP 工具（零新增依赖）。

目录结构：
- registry.py     管控层：参数校验 / 身份越权 / 超时隔离 / 审计脱敏
- tools.py        工具实现：HTTP 薄壳调常驻 FastAPI 服务（模型预热在服务侧，工具永不冷加载）
- server_stdio.py 最小 MCP stdio Server（换行分隔 JSON-RPC 2.0，stdout 独占为协议通道）
- client.py       子进程客户端（握手 + tools/list + tools/call）

安全边界：MCP server 只做转发，不直接连库、不加载模型；所有调用带 Bearer Token
走常驻服务现有鉴权/审计/配额体系。工具不可用时返回明确错误，绝不 mock 数据。
"""
