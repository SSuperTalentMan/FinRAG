#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/orchestrator/graph.py — AnswerGraph 统一编排图 + run/astream 包装。

路由条件边把 question 分发到 rag / nl2sql / chitchat / multimodal / review：
- rag：文档知识库检索 + 精排 + 生成（融合前主链路能力）；
- nl2sql：结构化问数（融合 ChatBI）；
- chitchat：寒暄直答，不检索；
- multimodal：读图/识别类问题，走上传引导（真正的解析在 /review 上传流水线，
  融合 DocAudit），不再静默并入 rag 造成答非所问；
- review：文档合规审查（融合 DocAudit）—— 对话内直接受理，建任务后台跑流水线。

复用 DocAudit 的 astream(updates) 节点级进度模式，直通 SSE。
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from langgraph.graph import END, StateGraph

from config import get_config
from rag_qa.orchestrator.agentic import agentic_node
from rag_qa.orchestrator.nodes import (
    chitchat_node,
    multimodal_node,
    nl2sql_node,
    rag_answer_node,
    rag_retrieve_node,
    review_node,
)
from rag_qa.orchestrator.route import route_node
from rag_qa.orchestrator.state import AnswerState, new_state

_graph = None


def _branch(state: AnswerState) -> str:
    """条件边：依据 route 节点写入的 skill 分发。

    multimodal 不再并入 rag——读图类问题走纯文本检索只会答非所问，
    改由 multimodal_node 给出上传引导（真正的解析在 /review 上传流水线）。

    Agentic RAG（agentic_mode=true）时，rag/nl2sql 分支改走 Agent 循环
    （多跳检索 + 跨工具组合，失败在 agentic_node 内部回退固定路由）；
    chitchat/multimodal/review 不需要工具循环，仍走固定节点。
    """
    skill = state.get("skill", "rag")
    if skill in ("rag", "nl2sql") and get_config().orchestrator.agentic_mode:
        return "agentic"
    return skill


def build_answer_graph():
    global _graph
    if _graph is not None:
        return _graph
    g = StateGraph(AnswerState)
    g.add_node("route", route_node)
    g.add_node("rag_retrieve", rag_retrieve_node)
    g.add_node("rag_answer", rag_answer_node)
    g.add_node("nl2sql", nl2sql_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("multimodal", multimodal_node)
    g.add_node("review", review_node)
    g.add_node("agentic", agentic_node)

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route", _branch, {
            "rag": "rag_retrieve",
            "nl2sql": "nl2sql",
            "chitchat": "chitchat",
            "multimodal": "multimodal",
            "review": "review",
            "agentic": "agentic",
        }
    )
    g.add_edge("rag_retrieve", "rag_answer")
    g.add_edge("rag_answer", END)
    g.add_edge("nl2sql", END)
    g.add_edge("chitchat", END)
    g.add_edge("multimodal", END)
    g.add_edge("review", END)
    _graph = g.compile()
    return _graph


def get_answer_graph():
    """返回 AnswerGraph 编译实例（单例）。"""
    return build_answer_graph()


async def run_ask(
    question: str,
    role: str = "user",
    session_id: str = "",
    user_id: int | None = None,
    domain: str = "",
    username: str = "",
) -> AnswerState:
    """非流式：invoke AnswerGraph 并返回最终 State（含 answer/sources/table/skill）。"""
    st = time.monotonic()
    state = new_state(question, role=role, session_id=session_id, user_id=user_id,
                      domain=domain, username=username)
    graph = get_answer_graph()
    final = await graph.ainvoke(state)
    final["latency_ms"] = int((time.monotonic() - st) * 1000)
    final.setdefault("status", "ok")
    final.setdefault("degraded", False)
    return final


async def stream_ask(
    question: str,
    role: str = "user",
    session_id: str = "",
    user_id: int | None = None,
    domain: str = "",
    username: str = "",
) -> AsyncIterator[dict]:
    """流式：astream(updates) 逐节点产出进度事件，供 SSE 直通。

    产出 dict 结构：{"event": <节点名|done|error>, "data": {...}}。
    astream 每步给出节点 update（增量 map），这里累积 merged 以便 done 事件携带最终答案。
    """
    merged: dict = {}
    state = new_state(question, role=role, session_id=session_id, user_id=user_id,
                      domain=domain, username=username)
    graph = get_answer_graph()
    try:
        async for chunk in graph.astream(state, stream_mode="updates"):
            node = next(iter(chunk))
            data = chunk[node] or {}
            merged.update({k: v for k, v in data.items() if v})
            emit = {"skill": merged.get("skill", ""), "stage": data.get("stage", node)}
            if merged.get("sources"):
                emit["sources"] = merged["sources"]
            if merged.get("table"):
                emit["table"] = merged["table"]
            if merged.get("task"):
                emit["task"] = merged["task"]
            yield {"event": node, "data": emit}
        yield {"event": "done", "data": {
            "answer": merged.get("answer", ""),
            "skill": merged.get("skill", ""),
            "sources": merged.get("sources", []),
            "table": merged.get("table"),
            "task": merged.get("task"),
            "status": merged.get("status", "ok"),
        }}
    except asyncio.CancelledError:  # 客户端断流
        raise
    except Exception as e:  # noqa: BLE001
        yield {"event": "error", "data": {"message": str(e)[:200]}}