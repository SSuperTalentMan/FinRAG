#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/eval_nl2sql.py — 问数（ChatBI 融合）能力评测。

数据：evals/datasets/nl2sql_eval.jsonl（代表性业务问句 + 期望业务口径）。
方法：对每题跑 Nl2SqlService.ask() 真实管线（Schema 召回 → SQL 生成 →
SQLGuard → 只读执行），统计：
  - 可执行力 executable_rate：SQL 成功执行并返回结果(answer_type=='data') 的占比；
  - 口径正确率 caliber_correct_rate：
      * 全量口径：所有题（含拒答）由 LLM 判定结果是否恰当；
      * in-schema 口径：仅统计 data 题（把"库内确无数据"的正确拒答移出分母，
        更真实反映生成质量）。拒答仅当库内确实不存在该业务对象才算恰当；
  - 逐题明细：每题打印 answer_type、口径判分 correct/reason，便于定位低分题。
  - 平均延迟、缓存命中率。
门禁：executable_rate >= 0.80。

需业务库 chatbi_meta/biz_demo + Redis + LLM 可用；不可用时打印 SKIPPED 并退出 2
（不计入失败，仅提示需在联调环境补跑）。

用法：
    D:\\FinRag\\.venv\\Scripts\\python.exe scripts/eval_nl2sql.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GATE_EXECUTABLE = 0.80
DS_PATH = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "nl2sql_eval.jsonl"


def _load() -> list[dict]:
    rows: list[dict] = []
    for line in DS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def _judge_caliber(question: str, expected_intent: str, sql: str, text: str,
                         answer_type: str):
    """LLM 判分：结果是否恰当命中期望业务口径。不可用返回 (None, reason)。

    - answer_type=='data'：判断 SQL/结果是否真正回答了期望业务口径；
    - answer_type=='refused'：判断拒答是否合理（库内确实不存在该指标所需的
      业务对象/字段才算合理，否则判错）。
    """
    try:
        from services.llm_ext import chat_json
        from config import get_config

        cfg = get_config()
        if answer_type == "refused":
            sys_p = ('你是评测员。下面是一道「问数」问题，系统返回了「拒答」'
                     '（业务库中查不到数据）。请判断该拒答是否合理。\n'
                     '判定标准：只有当业务库中确实不存在该指标所需的业务对象/字段'
                     '（例如问「毛利率」但库里没有利润/成本表、问「库存」但库里没有'
                     '库存表）时，拒答才算合理(correct=true)；若库里明明有相关表却'
                     '拒答，则算错误(correct=false)。\n'
                     '只输出 JSON: {"correct": true 或 false, "reason": "简短说明"}')
        else:
            sys_p = ('你是评测员。判断下面「问数结果」是否真正回答了「期望业务口径」。\n'
                     '只输出 JSON: {"correct": true 或 false, "reason": "简短说明"}')
        user_p = (
            f"用户问题：{question}\n期望业务口径：{expected_intent}\n"
            f"生成SQL：{sql}\n返回结果摘要：{text[:500]}"
        )
        payload, _ = await chat_json(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": user_p}],
            model=cfg.llm.model, temperature=0.0, stage="eval.nl2sql",
        )
        return bool(payload.get("correct")), str(payload.get("reason", ""))
    except Exception as e:  # noqa: BLE001
        return None, f"LLM判分不可用: {e}"


async def main() -> int:
    data = _load()

    # 初始化真实问数管线；不可用时优雅跳过（避免把"联调环境没起"当失败）
    try:
        from rag_qa.nl2sql.meta_store import MetaStore
        from rag_qa.nl2sql.service import Nl2SqlService
        from rag_qa.orchestrator.route import get_route_meta

        meta = get_route_meta()
        await meta.ensure_loaded()
        svc = Nl2SqlService(meta)
        await svc.ensure_ready()
    except Exception as e:  # noqa: BLE001
        print(f"SKIPPED: 问数管线不可用（需业务库 chatbi_meta/biz_demo + Redis + LLM）: {e}")
        return 2

    # LLM Key 未注入（API_KEY/DASHSCOPE_API_KEY）时明确跳过，避免误报 FAIL
    from config import get_config as _gc

    if not _gc().llm.api_key:
        print("SKIPPED: LLM API Key 未注入（设置 API_KEY/DASHSCOPE_API_KEY 后重跑）")
        return 2

    total = len(data)
    executed = 0
    refused = 0
    errored = 0
    cached = 0
    latencies: list[float] = []

    # 逐题记录
    records: list[dict] = []
    # 口径判分容器
    caliber_full: list[bool | None] = []      # 全量（含拒答）
    caliber_inschema: list[bool | None] = []  # 仅 data 题

    for it in data:
        q = it["question"]
        t0 = time.monotonic()
        try:
            ans = await svc.ask(q, role="user")
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {q}  {str(e)[:120]}")
            latencies.append(time.monotonic() - t0)
            records.append({"q": q, "type": "error", "rows": 0,
                            "caliber": None, "reason": str(e)[:80]})
            errored += 1
            continue
        dt = time.monotonic() - t0
        latencies.append(dt)
        at = ans.answer_type
        if at == "data":
            executed += 1
        elif at == "refused":
            refused += 1
        else:
            errored += 1
        if getattr(ans, "cached", False):
            cached += 1
        c, reason = await _judge_caliber(
            q, it.get("expected_intent", ""), ans.sql or "", ans.text or "", at
        )
        caliber_full.append(c)
        if at == "data":
            caliber_inschema.append(c)
        records.append({"q": q, "type": at, "rows": getattr(ans, "row_count", 0),
                        "caliber": c, "reason": reason,
                        "cached": getattr(ans, "cached", False), "lat": dt})
        cal_mark = "✓" if c is True else ("✗" if c is False else "?")
        print(f"  [{at}] {cal_mark} {q}  rows={getattr(ans, 'row_count', 0)} "
              f"cached={getattr(ans, 'cached', False)} lat={dt:.1f}s  {reason[:60]}")

    exec_rate = executed / total if total else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    def _rate(buckets):
        valid = [c for c in buckets if c is not None]
        return (sum(1 for c in valid if c) / len(valid)) if valid else None

    cal_full = _rate(caliber_full)
    cal_in = _rate(caliber_inschema)

    print("=" * 72)
    print(f"问数评测数据集: {total} 题")
    print(f"可执行力: {exec_rate:.2%} (门禁 {GATE_EXECUTABLE:.0%}) | "
          f"data={executed} refused={refused} error={errored}")
    if cal_in is not None:
        print(f"口径正确率(in-schema,仅 {executed} 道 data): {cal_in:.2%}")
    if cal_full is not None:
        print(f"口径正确率(全量 {total} 题): {cal_full:.2%}（含拒答，拒答恰当不计错）")
    print(f"平均延迟: {avg_lat:.2f}s | 缓存命中: {cached}/{total}")
    print("-" * 72)
    print("逐题明细:")
    for r in records:
        cm = "✓" if r["caliber"] is True else ("✗" if r["caliber"] is False else "·")
        print(f"  {cm} [{r['type']:7}] {r['q']}  rows={r['rows']}  {r['reason'][:70]}")
    print("=" * 72)

    ok = exec_rate >= GATE_EXECUTABLE
    print("PASS: 可执行力达到门禁" if ok else "FAIL: 可执行力低于门禁")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
