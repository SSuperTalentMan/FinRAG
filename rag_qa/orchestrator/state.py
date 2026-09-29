#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/orchestrator/state.py — AnswerGraph 统一状态定义。

所有 Skill（rag / nl2sql / chitchat / multimodal / review）共用一份 State，
路由节点写入 skill，各 Skill 节点回填 answer / sources / table / task，
编排层 run() 把最终 State 序列化给路由层组成统一响应。
"""
from __future__ import annotations

from typing import Any, TypedDict


class AnswerState(TypedDict, total=False):
    # ── 输入上下文 ──
    question: str            # 用户原始问题
    role: str                # 当前用户角色（user / admin / viewer ...）
    session_id: str          # 会话 ID（多轮历史）
    user_id: int | None      # 用户 ID（配额/落盘）
    username: str            # 用户名（审查任务归属 / 审计留痕）
    domain: str              # RAG 领域（final 由 classify_intent 填充）

    # ── 路由输出 ──
    skill: str               # rag | nl2sql | chitchat | multimodal | review
    stage: str               # 当前节点进度（SSE 桥用）
    route_info: dict         # 路由依据（规则/实体得分/修正原因），诊断与评估用

    # ── 文档审查（DocAudit 长任务）──
    task: dict | None        # {task_id, status, doc_name} —— 对话内拉起的审查任务

    # ── RAG / 通用答案 ──
    context: str             # 检索到的上下文文本
    sources: list[dict]      # 来源（question/score/source/source_url）

    # ── NL2SQL ──
    sql: str                 # 生成的 SQL
    table: dict | None       # {columns, rows, row_count, tables}

    # ── 输出 ──
    answer: str
    status: str              # ok | degraded | refused | error
    degraded: bool
    error: str
    meta: dict[str, Any]     # 附注（assumptions/repair_rounds/latency_ms ...）
    latency_ms: int
    trace: list[str]         # 节点执行轨迹（可观测/评估）


def new_state(
    question: str,
    role: str = "user",
    session_id: str = "",
    user_id: int | None = None,
    domain: str = "",
    username: str = "",
) -> AnswerState:
    """构造 AnswerGraph 初始 State（默认值统一收口到此处）。"""
    return {
        "question": question,
        "role": role or "user",
        "session_id": session_id or "",
        "user_id": user_id,
        "domain": domain or "",
        "username": username or "",
        "skill": "rag",
        "stage": "init",
        "route_info": {},
        "context": "",
        "sources": [],
        "sql": "",
        "table": None,
        "task": None,
        "answer": "",
        "status": "ok",
        "degraded": False,
        "error": "",
        "meta": {},
        "latency_ms": 0,
        "trace": ["init"],
    }