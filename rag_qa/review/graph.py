#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/graph.py — LangGraph 审查流水线（融合 DocAudit graph）。

prepare → review_batch（检索+比对+双温度仲裁+分级）→ advance（条件边循环）→ report。
`graph.stream()` 按节点产出进度，直通 SSE。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from config import get_config
from rag_qa.review.prompts import COMPARE_AGENT, GRADE_AGENT
from rag_qa.review.schemas import ClauseReviewOut, CompareOutModel, GradeOutModel, ViolatedRule
from rag_qa.review.rule_store import RuleRetriever
from services.llm_ext import chat_json_validated

logger = logging.getLogger(__name__)


class ReviewState(TypedDict):
    clauses: list[dict]
    cursor: int
    batch_size: int
    results: list[dict]
    summaries: list[dict]


def _fallback_risk(verdict: str, hit_severities: set[str]) -> str:
    """分级兜底：按命中规则的严重度确定性映射（无 LLM 参与时的下限保障）。"""
    if verdict != "violation":
        return "low"
    if "high" in hit_severities:
        return "high"
    if "medium" in hit_severities or hit_severities:
        return "medium"
    return "medium"


def _has_more(cursor: int, total: int, batch_size: int) -> bool:
    return cursor < total


def _rule_lines(rules: list[dict]) -> str:
    lines = []
    for i, r in enumerate(rules, 1):
        lines.append(f"{i}. [{r['rule_id']}] {r['doc_name']} {r.get('article_no') or ''}: {r['requirement']}")
    return "\n".join(lines) if lines else "(未检索到相关规则)"


async def _review_one(clause: dict, retriever: RuleRetriever) -> dict:
    """单条款：检索 → 比对 Agent（Pydantic 校验）→ 双温度仲裁 → 分级 Agent。"""
    cfg = get_config()
    query = f"{clause.get('title', '')}\n{clause['content']}".strip()
    rules, debug = await retriever.retrieve(query)

    compare_messages = [
        {"role": "system", "content": COMPARE_AGENT},
        {"role": "user", "content":
            f"【待审条款】第?条 {clause.get('title', '')}\n{clause['content']}\n\n【监管规则】\n{_rule_lines(rules)}"},
    ]
    try:
        compare_model, _ = await chat_json_validated(
            compare_messages, model=cfg.llm.sql_model or None, schema=CompareOutModel, stage="compare")
        compare_out = compare_model.model_dump()
    except Exception as e:  # noqa: BLE001
        logger.warning("compare failed clause=%s: %s", clause.get("clause_no"), e)
        compare_out = {"verdict": "insufficient_evidence", "violated_rules": [],
                       "evidence": f"审查模型调用失败: {e}", "suggestion": ""}

    verdict = compare_out.get("verdict", "insufficient_evidence")
    if verdict not in ("compliant", "violation", "insufficient_evidence"):
        verdict = "insufficient_evidence"
    vrules_raw = compare_out.get("violated_rules") or []
    valid_rule_ids = {r["rule_id"] for r in rules}
    vrules = [
        ViolatedRule(
            rule_id=str(v.get("rule_id", "")), doc_name=str(v.get("doc_name", "")),
            article_no=str(v.get("article_no", "")), evidence_quote=str(v.get("evidence_quote", ""))[:300],
        )
        for v in vrules_raw if isinstance(v, dict) and v.get("rule_id") in valid_rule_ids
    ] if verdict == "violation" else []

    # 双温度仲裁：violation 属高代价结论，用温度 0.7 复核一次，冲突即转人工
    arbitration_note = ""
    if cfg.compliance.dual_temp and verdict == "violation":
        try:
            arb_model, _ = await chat_json_validated(
                compare_messages, model=cfg.llm.sql_model or None, schema=CompareOutModel,
                temperature=0.7, stage="compare_arbitration")
            if arb_model.verdict != "violation":
                arbitration_note = (
                    f"双温度仲裁:温度0.7复核判定为 {arb_model.verdict},与首判冲突,转人工确认")
        except Exception as e:  # noqa: BLE001
            logger.warning("arbitration failed clause=%s: %s", clause.get("clause_no"), e)

    severity_of = {r["rule_id"]: r.get("severity", "medium") for r in rules}
    hit_severities = {severity_of.get(v.rule_id, "medium") for v in vrules}
    # 把条款原文摘要一并交给分级 Agent：合规但含财务责任（违约金/罚金/赔偿）的条款
    # 须判 medium，否则 GRADE 仅凭 verdict 无法区分「标准条款」与「带罚则条款」。
    grade_in = {
        "verdict": verdict,
        "rule_severities": sorted(hit_severities),
        "clause_excerpt": (clause.get("content") or clause.get("title") or "")[:200],
    }
    try:
        grade_model, _ = await chat_json_validated(
            [{"role": "system", "content": GRADE_AGENT},
             {"role": "user", "content": json.dumps(grade_in, ensure_ascii=False)}],
            model=cfg.llm.fast_model or None, schema=GradeOutModel, stage="grade")
        grade_out = grade_model.model_dump()
    except Exception:  # noqa: BLE001
        grade_out = {}

    risk = grade_out.get("risk_level", "")
    if risk not in ("high", "medium", "low"):
        risk = _fallback_risk(verdict, hit_severities)
    needs_human = (bool(grade_out.get("needs_human")) or risk == "high" or bool(arbitration_note))

    evidence = str(compare_out.get("evidence", ""))[:800]
    if arbitration_note:
        evidence = f"{evidence} | {arbitration_note}"[:800]
    out = ClauseReviewOut(
        clause_no=clause["clause_no"], clause_id=clause["clause_id"],
        verdict=verdict, risk_level=risk,
        violated_rules=[v.model_dump() for v in vrules],
        evidence=evidence,
        suggestion=str(compare_out.get("suggestion", ""))[:500],
        hitl_status="pending_review" if needs_human and verdict == "violation" else None,
    ).model_dump()
    out["_debug"] = debug
    return out


def build_graph(retriever: RuleRetriever):
    """构建审查 StateGraph；graph.stream() 可按节点产出进度。"""

    async def prepare(state: ReviewState) -> dict:
        return {"cursor": 0, "results": [], "summaries": []}

    async def review_batch(state: ReviewState) -> dict:
        start = state["cursor"]
        batch = state["clauses"][start: start + state["batch_size"]]
        results = list(state["results"])
        summaries = list(state["summaries"])
        outs = await asyncio.gather(*(_review_one(c, retriever) for c in batch))
        for clause, out in zip(batch, outs):
            results.append(out)
            summaries.append({"clause_no": clause["clause_no"], "debug": out["_debug"]})
        return {"results": results, "summaries": summaries,
                "progress": {"done": len(results), "total": len(state["clauses"])}}

    def route(state: ReviewState) -> str:
        if _has_more(state["cursor"], len(state["clauses"]), state["batch_size"]):
            return "review_batch"
        return "report"

    async def advance(state: ReviewState) -> dict:
        return {"cursor": state["cursor"] + state["batch_size"]}

    async def report(state: ReviewState) -> dict:
        return {"done": True}

    g = StateGraph(ReviewState)
    g.add_node("prepare", prepare)
    g.add_node("review_batch", review_batch)
    g.add_node("advance", advance)
    g.add_node("report", report)
    g.set_entry_point("prepare")
    g.add_edge("prepare", "review_batch")
    g.add_edge("review_batch", "advance")
    g.add_conditional_edges("advance", route, {"review_batch": "review_batch", "report": "report"})
    g.add_edge("report", END)
    return g.compile()


async def run_review(clauses: list[dict], retriever: RuleRetriever):
    """执行审查；按节点产出进度事件（兼容 langgraph 不同版本 astream 形态）。"""
    cfg = get_config()
    graph = build_graph(retriever)
    init: ReviewState = {
        "clauses": clauses, "cursor": 0, "batch_size": cfg.compliance.batch_size,
        "results": [], "summaries": [],
    }
    results: list[dict] = []
    async for item in graph.astream(init, stream_mode="updates"):
        if isinstance(item, tuple) and len(item) == 2:
            node, update = item
        elif isinstance(item, dict):
            node, update = next(iter(item.items()))
        else:
            continue
        update = update or {}
        if node == "review_batch":
            if update.get("results") is not None:
                results = update["results"]
            if update.get("progress"):
                yield "review_batch", {"progress": update["progress"], "results": list(results)}
        elif node == "report":
            yield "report", {"results": list(results)}