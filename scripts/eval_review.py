#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/eval_review.py — 文档合规审查（DocAudit 融合）能力评测。

数据：evals/datasets/review_eval.jsonl（金标条款 + 期望 verdict/risk_level）。
方法：把金标条款拼成一份合成合同，经 review_bridge.submit_inline_review 拉起
真实审查流水线；轮询 docaudit_clause_review 直到产出或超时；按条款正文对齐金标，
统计：
  - 条款召回 clause_recall：金标条款被抽取到的占比；
  - 判定准确率 judgment_accuracy：verdict 与 risk_level 同时与金标一致的占比。
门禁：clause_recall >= 0.70 且 judgment_accuracy >= 0.80。

需 MySQL + 审查流水线 + LLM 可用；不可用时打印 SKIPPED 并退出 2。
注意：会真实创建审查任务并跑流水线（含可能的 HITL 人工复核），仅在联调环境执行。

用法：
    D:\\FinRag\\.venv\\Scripts\\python.exe scripts/eval_review.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GATE_RECALL = 0.70
GATE_ACCURACY = 0.80
POLL_TIMEOUT = 300.0
POLL_INTERVAL = 5.0
DS_PATH = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "review_eval.jsonl"


def _load() -> list[dict]:
    rows: list[dict] = []
    for line in DS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _build_contract(golds: list[dict]) -> str:
    """把金标条款拼成带条号的分页合同文本，供 DocAudit 解析链路消费。"""
    lines = ["合成评测合同", "=" * 20]
    for i, g in enumerate(golds, 1):
        lines.append(f"第{i}条 {g['clause']}")
    return "\n".join(lines)


def _norm(v: str) -> str:
    return (v or "").strip().lower()


async def _fetch_reviews(task_id: int) -> list[dict]:
    """读取该任务的条款审查结果（含条款正文，用于与金标对齐）。"""
    from rag_qa.review.db import afetch_all

    return await afetch_all(
        "SELECT cr.clause_no, cr.verdict, cr.risk_level, c.content "
        "FROM docaudit_clause_review cr "
        "JOIN docaudit_clause c ON c.id = cr.clause_id "
        "WHERE cr.task_id = %s",
        (task_id,),
    )


async def _poll(task_id: int) -> list[dict]:
    """轮询审查产出，直到有结果或超时。"""
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        try:
            rows = await _fetch_reviews(task_id)
        except Exception as e:  # noqa: BLE001
            print(f"  [poll-error] {str(e)[:120]}")
            rows = []
        if rows:
            return rows
        await asyncio.sleep(POLL_INTERVAL)
    return []


async def main() -> int:
    golds = _load()
    try:
        from rag_qa.orchestrator.review_bridge import submit_inline_review
    except Exception as e:  # noqa: BLE001
        print(f"SKIPPED: 审查桥接不可用: {e}")
        return 2

    contract = _build_contract(golds)

    # LLM Key 未注入时明确跳过，避免误报 FAIL
    try:
        from config import get_config as _gc

        if not _gc().llm.api_key:
            print("SKIPPED: LLM API Key 未注入（设置 API_KEY/DASHSCOPE_API_KEY 后重跑）")
            return 2
    except Exception:  # noqa: BLE001
        pass

    try:
        task_id, doc_name = await submit_inline_review(contract, user="eval")
    except Exception as e:  # noqa: BLE001
        print(f"SKIPPED: 审查管线不可用（需 MySQL + 审查流水线 + LLM）: {e}")
        return 2

    print(f"评测任务已创建 task={task_id} doc={doc_name}，轮询审查产出…")
    reviews = await _poll(task_id)
    if not reviews:
        print("FAIL: 超时未产出审查结果")
        return 1

    # 按条款正文对齐金标
    matched = 0
    correct = 0
    for g in golds:
        gtext = g["clause"]
        hit = next((r for r in reviews if gtext in (r.get("content") or "")
                    or (r.get("content") or "") in gtext), None)
        if hit is None:
            continue
        matched += 1
        ok_verdict = _norm(hit.get("verdict")) == _norm(g["expected_verdict"])
        ok_risk = _norm(hit.get("risk_level")) == _norm(g["expected_risk_level"])
        if ok_verdict and ok_risk:
            correct += 1
        else:
            print(f"  [mismatch] {gtext[:30]}… 期望={g['expected_verdict']}/{g['expected_risk_level']} "
                  f"实际={hit.get('verdict')}/{hit.get('risk_level')}")

    recall = matched / len(golds) if golds else 0.0
    acc = correct / matched if matched else 0.0

    print("=" * 60)
    print(f"审查评测数据集: {len(golds)} 条金标条款 | 任务 {task_id}")
    print(f"条款召回: {recall:.2%} (门禁 {GATE_RECALL:.0%}) | 命中 {matched}/{len(golds)}")
    print(f"判定准确率: {acc:.2%} (门禁 {GATE_ACCURACY:.0%}) | 正确 {correct}/{matched}")
    print("=" * 60)

    ok = recall >= GATE_RECALL and acc >= GATE_ACCURACY
    print("PASS: 审查能力达到门禁" if ok else "FAIL: 审查能力低于门禁")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
