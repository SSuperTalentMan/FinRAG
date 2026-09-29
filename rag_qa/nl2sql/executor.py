#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/executor.py — 只读执行器（纵深防御第三层）。

流程：业务库只读连接 → 会话级 READ ONLY + max_execution_time（覆盖 CTE 等一切语句形态）
→ EXPLAIN 预检（暴露语法/字段错误、估算扫描行数）→ SELECT 挂查询超时 hint → 执行。
"""
from __future__ import annotations

import asyncio
import datetime
import decimal
from typing import Any

from config import get_config
from rag_qa.nl2sql import db


class GuardReject(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _json_safe(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(round(v, 4))
    if isinstance(v, datetime.datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, datetime.date):
        return v.isoformat()  # date.isoformat() 不接受 sep/timespec 关键字参数
    if isinstance(v, datetime.timedelta):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return v


def _with_timeout_hint(sql: str, timeout_ms: int) -> str:
    """SELECT /*+ MAX_EXECUTION_TIME(ms) */ ... —— hint 只能挂 SELECT 开头；
    CTE 由会话级 max_execution_time 兜底。"""
    lower = sql.lstrip()
    if lower.lower().startswith("select"):
        idx = sql.lower().find("select") + len("select")
        return f"{sql[:idx]} /*+ MAX_EXECUTION_TIME({timeout_ms}) */{sql[idx:]}"
    return sql


def _run_sql_sync(sql: str, timeout_ms: int, scan_threshold: int) -> tuple[list[str], list[list[Any]]]:
    def _work(conn):
        nonlocal total_rows
        cur = conn.cursor()
        # 会话级只读 + 超时，覆盖 CTE/Union 等 hint 覆盖不到的形态
        cur.execute("SET SESSION TRANSACTION READ ONLY")
        cur.execute(f"SET SESSION max_execution_time = {int(timeout_ms)}")
        # EXPLAIN 预检
        cur.execute(f"EXPLAIN {sql}")
        for r in cur.fetchall():
            rows = r.get("rows")
            if isinstance(rows, int):
                total_rows += rows
        if total_rows > scan_threshold:
            raise GuardReject(f"预计扫描行数过大({total_rows}),建议按维度聚合或缩小时间范围")
        # 执行已通过 Guard 的 SELECT
        cur.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        rows_out = [[_json_safe(v) for v in row.values()] for row in cur.fetchall()]
        return columns, rows_out

    total_rows = 0
    # 走 db.with_biz_conn：连接来自池，finally 归还（池满时阻塞等待，不打爆 MySQL）
    return db.with_biz_conn(_work)


async def run_readonly_sql(
    sql: str,
    allowed_tables: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """执行经过 Guard 校验的只读 SELECT（在业务库只读连接上）。

    allowed_tables：当前角色可见表白名单。传入时在执行前**再校验一次**——
    把 RBAC 变成执行器的内在约束，而不是"指望调用方记得先跑 sqlguard"。
    任何绕过 Guard 直接调本函数的路径（脚本/未来编排/测试）都不会因此越权读表。
    """
    cfg = get_config().nl2sql
    if allowed_tables is not None:
        from rag_qa.nl2sql import sqlguard

        verdict = sqlguard.check(sql, set(allowed_tables))
        if not verdict.ok:
            raise GuardReject(f"SQLGuard 执行前复核未通过: {verdict.reject_reason}")
    hinted = _with_timeout_hint(sql, cfg.timeout_ms)
    return await asyncio.to_thread(
        _run_sql_sync, hinted, cfg.timeout_ms, cfg.explain_scan_threshold,
    )