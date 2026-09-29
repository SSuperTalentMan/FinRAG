#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/orchestrator/agentic.py — Agentic RAG 节点（P4）。

与固定编排（route -> rag_retrieve -> rag_answer）的区别：
固定路由一次定型，问数与问答互斥；本节点把「工具列表」交给 LLM 循环决策——
  规划 → 调工具（kb_search / sql_query）→ 观察结果 → 不够就换问法再调 → 收尾作答。
由此获得三个固定编排做不到的能力：
  1. 多跳检索：第一轮召回不相关时自主改写重查（最多 agentic_max_hops 跳）；
  2. 跨工具组合：「存款保险最高赔多少，再查下今年赔付笔数」可串 RAG + 问数两个工具；
  3. 结果自评：观察命中数/分数后决定继续检索还是收尾。

安全设计（金融场景确定性优先）：
- 配置开关 agentic_mode 默认关闭；关闭时 graph 走原固定路由，行为零变化；
- LLM 决策失败 / 超过跳数 / 工具连续异常 → 一律回退固定路由节点（rag/nl2sql），
  绝不把"Agent 失败"当答案返回；
- 工具就是进程内的检索与问数服务（与固定路由同一份代码），无新增外部依赖与权限面。
"""
from __future__ import annotations

import asyncio
import logging

from config import get_config
from rag_qa.core.query_classifier import classify_intent
from rag_qa.orchestrator.state import AnswerState

logger = logging.getLogger(__name__)

# 复用与固定路由完全相同的底层实现（行为一致性优先）
from routers.chat import _build_context, _load_recent_history
from services.llm import generate_answer, chat_completion

_DECISION_SYSTEM = (
    "你是 FinRag 的检索规划器。可用工具：\n"
    "1. kb_search —— 金融知识库检索（监管政策/会计准则/业务释义），"
    "返回相关片段列表；\n"
    "2. sql_query —— 经营数据问数（销售额/订单/退货率/客单价/同比环比等结构化数据）；\n"
    "3. finish —— 已收集足够资料，生成最终答案。\n"
    "策略：优先判断问题需要【知识】还是【业务数据】还是【两者都要】；"
    "两者都要时先 kb_search 再 sql_query（或反过来）各调一次。"
    "观察每轮结果：命中不足或明显偏题时，换一个更精确的问法再检索（不要重复同一问法）；"
    "资料足够立即 finish，不要空转。\n"
    '只输出 JSON: {"action": "kb_search|sql_query|finish", "query": "本轮检索/问数用的问题", '
    '"reason": "一句话理由"}'
)

_OBS_MAX = 400  # 单条观察回喂给 LLM 的最大长度（控制上下文膨胀）


def _trim(s: str, n: int = _OBS_MAX) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


async def _decide(question: str, hops: list[dict]) -> dict | None:
    """让 LLM 决策下一跳；失败返回 None（触发回退）。"""
    try:
        from services.llm_ext import chat_json

        cfg = get_config()
        model = cfg.orchestrator.route_model or cfg.llm.model
        lines = [f"用户问题：{question}"] + [
            f"第{i + 1}跳[{h['action']}] q={h['query']} -> {_trim(h['observation'], 200)}"
            for i, h in enumerate(hops)
        ]
        payload, _ = await chat_json(
            [
                {"role": "system", "content": _DECISION_SYSTEM},
                {"role": "user", "content": "\n".join(lines)},
            ],
            model=model, temperature=0.0, stage="orchestrator.agentic",
        )
        action = str(payload.get("action", "")).strip().lower()
        if action not in ("kb_search", "sql_query", "finish"):
            return None
        return {
            "action": action,
            "query": str(payload.get("query", "") or question).strip(),
            "reason": str(payload.get("reason", ""))[:100],
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Agent 决策失败，回退固定路由: %s", str(e)[:150])
        return None


async def _tool_kb_search(query: str) -> tuple[str, list[dict], str]:
    """知识库检索（与 rag_retrieve_node 同一份 _build_context）。"""
    intent = await asyncio.to_thread(classify_intent, query)
    context, sources = await asyncio.to_thread(_build_context, query, intent, None)
    top_score = max((s.get("score") or 0) for s in sources) if sources else 0
    obs = f"命中 {len(sources)} 条，最高相关分 {top_score:.3f}；首条: " + _trim(
        (sources[0].get("question") or sources[0].get("source") or "") if sources else "无", 120
    )
    return context or "", sources, obs


async def _tool_sql_query(query: str, state: AnswerState) -> tuple[dict | None, str, str]:
    """问数（与 nl2sql_node 同一份 Nl2SqlService）。"""
    from rag_qa.orchestrator.nodes import _get_nl2sql
    from services.metrics import record_nl2sql

    svc = _get_nl2sql()
    await svc.ensure_ready()
    ans = await svc.ask(query, role=state.get("role", "user"))
    try:
        outcome = "data" if ans.answer_type == "data" else (ans.status or "failed")
        record_nl2sql(outcome, (ans.latency_ms or 0) / 1000.0)
    except Exception:  # noqa: BLE001
        pass
    if ans.answer_type == "data":
        table = {
            "columns": ans.columns,
            "rows": ans.rows,
            "row_count": ans.row_count,
            "tables": ans.tables,
        }
        obs = f"返回 {ans.row_count} 行数据，涉及表 {ans.tables}；摘要: {_trim(ans.text, 160)}"
        return table, ans.sql or "", obs
    return None, "", f"未返回数据（status={ans.status}）；原因: {_trim(ans.text, 160)}"


async def _fallback_fixed(state: AnswerState) -> AnswerState:
    """回退固定路由：按 route 已判定的 skill 执行原节点（行为与关闭 agentic 完全一致）。

    注意 rag_retrieve_node / rag_answer_node / nl2sql_node 都是 async 协程，
    必须 await —— 否则返回协程对象，LangGraph 报 "Expected dict, got coroutine"。
    """
    from rag_qa.orchestrator.nodes import nl2sql_node, rag_answer_node, rag_retrieve_node

    skill = state.get("skill", "rag")
    state["trace"].append(f"agentic.fallback->{skill}")
    if skill == "nl2sql":
        return await nl2sql_node(state)
    state = await rag_retrieve_node(state)
    return await rag_answer_node(state)


async def agentic_node(state: AnswerState) -> AnswerState:
    """Agentic RAG 节点：LLM 循环决策（kb_search/sql_query/finish），失败回退固定路由。"""
    question = state["question"]
    cfg = get_config().orchestrator
    max_hops = max(1, int(cfg.agentic_max_hops))

    hops: list[dict] = []
    kb_context = ""
    kb_sources: list[dict] = []
    table: dict | None = None
    sql_text = ""
    nl2sql_text = ""

    try:
        for _ in range(max_hops):
            decision = await _decide(question, hops)
            if decision is None:
                state["error"] = "agentic: LLM 决策失败"
                return await _fallback_fixed(state)

            action, query = decision["action"], decision["query"]
            if action == "finish":
                break
            if action == "kb_search":
                ctx, srcs, obs = await _tool_kb_search(query)
                kb_context = (kb_context + "\n\n" + ctx).strip() if kb_context else ctx
                kb_sources.extend(srcs[:3])
                hops.append({"action": action, "query": query, "observation": obs,
                             "reason": decision["reason"]})
            else:  # sql_query
                tb, sql, obs = await _tool_sql_query(query, state)
                if tb:
                    table, sql_text = tb, sql
                hops.append({"action": action, "query": query, "observation": obs,
                             "reason": decision["reason"]})
                if tb is None and not nl2sql_text:
                    nl2sql_text = obs  # 拒答/澄清时至少留个说明
        else:
            # 跳数用尽仍未 finish：视为决策未收敛，回退固定路由（确定性兜底）
            hops.append({"action": "exhausted", "query": "", "observation": f"达到最大跳数 {max_hops}",
                         "reason": ""})
            state["error"] = "agentic: 跳数用尽未收敛"
            return await _fallback_fixed(state)
    except Exception as e:  # noqa: BLE001 工具链路任何异常都回退固定路由
        logger.warning("Agentic 循环异常，回退固定路由: %s", str(e)[:150])
        state["error"] = f"agentic: {str(e)[:200]}"
        return await _fallback_fixed(state)

    # ── 收尾：按已收集的资料生成最终答案 ──────────────────────────────────────
    history = ""
    if state.get("session_id"):
        history = await asyncio.to_thread(
            _load_recent_history, state["session_id"], state.get("user_id")
        )

    try:
        if table and kb_context:
            # 跨工具组合：知识 + 数据一起给 LLM 综合作答
            table_brief = _trim(str(table.get("rows", [])[:5]), 600)
            prompt = (
                "基于以下两部分资料回答用户问题，分别说明知识依据与数据结论：\n\n"
                f"【知识库资料】\n{kb_context[:3000]}\n\n"
                f"【经营数据】SQL: {sql_text}\n结果(前5行): {table_brief}\n\n"
                f"用户问题：{question}"
            )
            answer = await asyncio.to_thread(chat_completion, [{"role": "user", "content": prompt}])
            state["sources"] = kb_sources
            state["table"] = table
            state["sql"] = sql_text
        elif table:
            answer = nl2sql_text or "问数完成，但未生成摘要文本。"
            state["table"] = table
            state["sql"] = sql_text
        elif kb_context:
            domain = state.get("domain") or ""
            answer = await asyncio.to_thread(
                generate_answer, question, kb_context, domain, history
            )
            state["sources"] = kb_sources
        else:
            # 什么都没收集到（如 finish 太早）：回退固定路由拿一个可靠答案
            state["error"] = "agentic: 未收集到任何资料"
            return await _fallback_fixed(state)
        state["status"] = "ok"
    except Exception as e:  # noqa: BLE001 生成失败降级：取检索首条做兜底
        from rag_qa.orchestrator.nodes import _extract_fallback_from_context

        answer = _extract_fallback_from_context(kb_context) or (
            "生成答案时服务暂时不可用，请稍后重试。"
        )
        state["status"] = "degraded"
        state["degraded"] = True
        state["error"] = str(e)[:300]
        state["sources"] = kb_sources

    state["answer"] = answer
    state["stage"] = "agentic"
    state["meta"] = {**(state.get("meta") or {}), "agent_hops": hops}
    state["trace"].extend(
        [f"agentic.{i + 1}:{h['action']}({_trim(h['query'], 40)})" for i, h in enumerate(hops)]
    )
    return state
