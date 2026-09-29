#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/mcp_layer/tools.py — FinRag MCP 工具实现（HTTP 薄壳，纯标准库）。

核心设计：**MCP server 只做转发，不加载模型、不连库**。
所有工具经 HTTP 调常驻 FastAPI 服务（默认 http://127.0.0.1:8000）——
BGE-M3/Reranker 的冷加载（约 1 分钟）由常驻服务承担，工具调用永不触发冷启动，
因此工具超时可以设得比较紧。服务不可达时返回明确错误，绝不 mock 数据。

鉴权：复用常驻服务现有 JWT 体系。Token 经环境变量注入（FINRAG_TOKEN），
不落盘、不进日志（registry 审计层会脱敏）。
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
import uuid

from rag_qa.mcp_layer.registry import Tool, ToolResult, ToolRegistry

_DEFAULT_BASE = "http://127.0.0.1:8000"
_QUESTION_MAX = 2000  # 与 ChatRequest.message 上限对齐


def _base_url() -> str:
    return os.environ.get("FINRAG_BASE_URL", _DEFAULT_BASE).rstrip("/")


def _token() -> str:
    return os.environ.get("FINRAG_TOKEN", "")


def _http_json(method: str, path: str, body: dict | None = None,
               timeout: float = 90.0) -> dict:
    """调常驻服务 JSON 接口。非 2xx / 不可达抛异常，由 handler 统一转 ToolResult。"""
    url = f"{_base_url()}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if _token():
        req.add_header("Authorization", f"Bearer {_token()}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_multipart(path: str, filename: str, content: bytes,
                    timeout: float = 60.0) -> dict:
    """multipart 文件上传（审查任务）。标准库手工拼 multipart/form-data。"""
    boundary = f"----finragmcp{uuid.uuid4().hex}"
    part = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    body = part + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(f"{_base_url()}{path}", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if _token():
        req.add_header("Authorization", f"Bearer {_token()}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wrap(fn):
    """把 handler 异常统一转成给人看懂的 ToolResult（服务不可达是最常见形态）。"""
    def inner(args: dict) -> ToolResult:
        try:
            return fn(args)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", "")
            except Exception:  # noqa: BLE001
                pass
            return ToolResult(False, error=f"FinRag 服务返回 {e.code}: {detail or e.reason}")
        except urllib.error.URLError as e:
            return ToolResult(
                False,
                error=f"FinRag 服务不可达（{_base_url()}）：{e.reason}。请确认服务已启动且 FINRAG_BASE_URL/FINRAG_TOKEN 配置正确。",
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(False, error=f"{type(e).__name__}: {str(e)[:200]}")
    return inner


def _extract(payload: dict) -> dict:
    """ApiResponse{success, data} → data；success=false 抛出带 message 的异常。"""
    if not payload.get("success", False):
        raise RuntimeError(str(payload.get("message", "服务返回 success=false"))[:200])
    return payload.get("data") or {}


# ══════════════════════ 工具 handler ══════════════════════════════════════════
@_wrap
def _h_ask(args: dict) -> ToolResult:
    """端到端统一问答：内部自动路由 rag/nl2sql/chitchat/review。"""
    data = _extract(_http_json(
        "POST", "/api/v1/chat/ask",
        {"message": args["question"], "session_id": args.get("session_id") or None},
    ))
    return ToolResult(True, data={
        "answer": data.get("answer", ""),
        "skill": data.get("skill", ""),
        "sources": (data.get("sources") or [])[:5],
        "table": data.get("table"),
        "status": data.get("status", ""),
    }, meta={"engine": "answer_graph"})


@_wrap
def _h_kb_search(args: dict) -> ToolResult:
    """只检索不生成：返回带溯源的块列表，供 Agent 自行判断是否够用。"""
    body = {"message": args["question"]}
    if args.get("top_k"):
        body["top_k"] = int(args["top_k"])
    data = _extract(_http_json("POST", "/api/v1/chat/search", body))
    sources = data.get("sources") or []
    return ToolResult(True, data={
        "hit_count": data.get("hit_count", len(sources)),
        "sources": sources[: int(args.get("top_k") or 5)],
    }, meta={"engine": "milvus_hybrid+bm25+rerank"})


@_wrap
def _h_sql_query(args: dict) -> ToolResult:
    """自然语言问数（NL2SQL + SQLGuard + 只读执行）。"""
    data = _extract(_http_json("POST", "/api/v1/nl2sql/ask", {"question": args["question"]}))
    if data.get("answer_type") == "data":
        rows = data.get("rows") or []
        return ToolResult(True, data={
            "text": data.get("text", ""),
            "sql": data.get("sql", ""),
            "columns": data.get("columns", []),
            "rows": rows[:20],
            "row_count": data.get("row_count", len(rows)),
            "tables": data.get("tables", []),
        }, meta={"engine": "nl2sql", "repair_rounds": data.get("repair_rounds")})
    return ToolResult(True, data={
        "text": data.get("text", ""),
        "answer_type": data.get("answer_type", ""),
        "status": data.get("status", ""),
    }, meta={"engine": "nl2sql", "note": "未返回数据（可能被拒答/需澄清），text 内有原因"})


@_wrap
def _h_review_upload(args: dict) -> ToolResult:
    """上传文档发起合同审查（长任务）：立即返回 task_id，用 finrag.review_status 轮询。"""
    filename = args.get("filename") or "upload.pdf"
    content = base64.b64decode(args["content_base64"])
    if not content:
        return ToolResult(False, error="文件内容为空")
    data = _extract(_http_multipart("/api/v1/review/upload", filename, content))
    task = data.get("task") or data
    return ToolResult(True, data={
        "task_id": task.get("task_id"),
        "doc_name": task.get("doc_name") or filename,
        "status": task.get("status", "uploaded"),
        "note": "审查是长任务（分钟级），请用 finrag.review_status 轮询，不要同步等待",
    }, meta={"engine": "docaudit"})


@_wrap
def _h_review_status(args: dict) -> ToolResult:
    """查询审查任务进度/结论。"""
    tid = args["task_id"]
    data = _extract(_http_json("GET", f"/api/v1/review/task/{tid}/status"))
    return ToolResult(True, data=data, meta={"engine": "docaudit"})


@_wrap
def _h_review_report(args: dict) -> ToolResult:
    """获取审查报告（任务完成后）。"""
    tid = args["task_id"]
    data = _extract(_http_json("GET", f"/api/v1/review/task/{tid}/report"))
    return ToolResult(True, data=data, meta={"engine": "docaudit"})


# ══════════════════════ 工具注册 ══════════════════════════════════════════════
_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "用户问题（自然语言，<=2000 字）"},
    },
    "required": ["question"],
}


def build_registry() -> ToolRegistry:
    """组装 FinRag MCP 工具注册表（6 个工具）。"""
    reg = ToolRegistry()
    reg.register(Tool(
        name="finrag.ask",
        description="FinRag 金融智能问答（端到端）：自动路由知识库问答/经营数据问数/闲聊/审查受理，返回带溯源的答案。不确定走哪条链路时先用这个。",
        input_schema={
            "type": "object",
            "properties": {
                "question": _QUESTION_SCHEMA["properties"]["question"],
                "session_id": {"type": "string", "description": "可选，多轮会话 ID"},
            },
            "required": ["question"],
        },
        handler=_h_ask,
        timeout=120.0,
    ))
    reg.register(Tool(
        name="finrag.kb_search",
        description="FinRag 知识库检索（只检索不生成）：返回知识库相关片段列表（含来源与相关分），供调用方判断资料是否够用或继续追问。",
        input_schema={
            "type": "object",
            "properties": {
                "question": _QUESTION_SCHEMA["properties"]["question"],
                "top_k": {"type": "integer", "description": "返回条数，默认 5"},
            },
            "required": ["question"],
        },
        handler=_h_kb_search,
        timeout=60.0,
    ))
    reg.register(Tool(
        name="finrag.sql_query",
        description="FinRag 经营数据问数：把自然语言转成 SQL 在只读连接执行，返回表格数据（销售额/订单/退货率/同比环比等）。",
        input_schema=_QUESTION_SCHEMA,
        handler=_h_sql_query,
        timeout=120.0,
    ))
    reg.register(Tool(
        name="finrag.review_upload",
        description="FinRag 合同审查上传：上传 PDF/图片发起合规审查长任务，立即返回 task_id（不要同步等待，用 finrag.review_status 轮询）。",
        input_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "文件名（带扩展名）"},
                "content_base64": {"type": "string", "description": "文件内容的 base64 编码"},
            },
            "required": ["filename", "content_base64"],
        },
        handler=_h_review_upload,
        timeout=90.0,
    ))
    reg.register(Tool(
        name="finrag.review_status",
        description="FinRag 审查任务进度查询：返回任务状态与审查结论摘要。",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "审查任务 ID"},
            },
            "required": ["task_id"],
        },
        handler=_h_review_status,
        timeout=30.0,
    ))
    reg.register(Tool(
        name="finrag.review_report",
        description="FinRag 审查报告获取：任务完成后返回完整审查报告数据。",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "审查任务 ID"},
            },
            "required": ["task_id"],
        },
        handler=_h_review_report,
        timeout=30.0,
    ))
    return reg
