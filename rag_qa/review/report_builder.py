#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/report_builder.py — 审查报告与摘要（含溯源完整性：每个违规结论必须携带 rule_id+页码）。"""
from __future__ import annotations

from datetime import datetime


def _page_span(page_start: int, page_end: int) -> str:
    p = f"第{page_start}" + (f"-{page_end}页" if page_end != page_start else "页")
    return p


def build_summary(results: list[dict]) -> dict:
    """合规总览：条款总数 / 合规 / 违规 / 证据不足 / 高风险 / 待人工复核数。"""
    total = len(results)
    verdicts = {"compliant": 0, "violation": 0, "insufficient_evidence": 0}
    high = 0
    hitl_pending = 0
    for r in results:
        v = r.get("verdict")
        if v in verdicts:
            verdicts[v] += 1
        if r.get("risk_level") == "high":
            high += 1
        if r.get("hitl_status") == "pending_review":
            hitl_pending += 1
    return {
        "total": total,
        "compliant": verdicts["compliant"],
        "violation": verdicts["violation"],
        "insufficient_evidence": verdicts["insufficient_evidence"],
        "high_risk": high,
        "hitl_pending": hitl_pending,
    }


def build_markdown(doc_name: str, summary: dict, results: list[dict],
                   clause_pages: dict[str, str], llm_digest: str = "") -> str:
    """生成 Markdown 审查报告。逐条款保留页码与规则溯源。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 合规审查报告\n",
        f"- 文档：{doc_name}",
        f"- 生成时间：{ts}",
        f"- 结论结构：条款总数 {summary['total']}｜合规 {summary['compliant']}｜违规 {summary['violation']}"
        f"｜证据不足 {summary['insufficient_evidence']}｜高风险 {summary['high_risk']}｜待复核 {summary['hitl_pending']}\n",
    ]
    if llm_digest:
        lines += ["## 执行摘要\n", llm_digest, "\n"]
    lines.append("## 风险条款详情\n")
    for r in results:
        if r.get("verdict") == "compliant":
            continue
        no = r.get("clause_no", "")
        pg = clause_pages.get(no, "")
        vr = r.get("violated_rules") or []
        rules = "、".join(f"{v.get('rule_id')}" for v in vr) or "（无）"
        lines += [
            f"### {no}（{pg}）",
            f"- 判定：{r.get('verdict')}｜风险：{r.get('risk_level')}",
            f"- 违反规则：{rules}",
            f"- 依据：{r.get('evidence', '')}",
            f"- 建议：{r.get('suggestion', '') or '无'}",
            "",
        ]
    lines.append("## 合规条款清单\n")
    for r in results:
        if r.get("verdict") == "compliant":
            no = r.get("clause_no", "")
            lines.append(f"- {no}（{clause_pages.get(no, '')}）")
    lines.append("\n> 报告由 LangGraph 多 Agent 审查流水线生成，结论均携带 rule_id + 页码溯源。")
    return "\n".join(lines)