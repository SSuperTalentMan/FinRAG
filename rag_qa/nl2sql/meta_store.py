#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/meta_store.py — 元数据访问层（table_registry / column_registry / metric_dict）。

进程内缓存元数据，提供：可用表、字段、DDL 文本构建、指标口径匹配。指标匹配用
"命中词最长者"去噪 —— 问"净销售额"时不应再注入 "SUM(销售额)" 的 GMV 口径。
"""
from __future__ import annotations

import asyncio
import threading
import time

from rag_qa.nl2sql import db
from rag_qa.nl2sql.schema import MetricInfo


class MetaStore:
    def __init__(self, watermark_ttl: int = 60):
        self.watermark_ttl = watermark_ttl
        self.table_registry_rows: list[dict] = []
        self._columns_by_table: dict[str, list[dict]] = {}
        self._metrics: list[dict] = []
        self._watermarks: dict[str, str] = {}
        self._watermark_ts: float = 0.0
        self._loaded = False
        self._lock = threading.Lock()

    def load_sync(self) -> None:
        """同步加载元数据（在 to_thread 或启动时调用）。"""
        self.table_registry_rows = db.fetch_all(
            "SELECT * FROM table_registry WHERE enabled = 1", meta=True)
        col_rows = db.fetch_all("SELECT * FROM column_registry", meta=True)
        self._columns_by_table = {}
        for r in col_rows:
            self._columns_by_table.setdefault(r["table_name"], []).append(r)
        self._metrics = db.fetch_all("SELECT * FROM metric_dict", meta=True)
        self._loaded = True

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    await asyncio.to_thread(self.load_sync)

    def enabled_tables(self) -> list[str]:
        return [r["table_name"] for r in self.table_registry_rows]

    def columns_of(self, table_name: str) -> list[dict]:
        return self._columns_by_table.get(table_name, [])

    def all_columns(self) -> list[dict]:
        return [c for cols in self._columns_by_table.values() for c in cols]

    def build_ddl_text(self, table_name: str) -> str:
        """为 LLM 构建「表名 + 用途 + 字段说明」上下文。"""
        trow = next((r for r in self.table_registry_rows if r["table_name"] == table_name), None)
        if not trow:
            return ""
        lines = [f"表 {table_name}({trow.get('display_name', '')}):{trow.get('description', '')}"]
        for c in self.columns_of(table_name):
            line = f"  - {c['column_name']} {c.get('data_type', '')}:{c.get('description', '')}"
            if c.get("sample_values"):
                line += f"(示例值: {c['sample_values']})"
            lines.append(line)
        return "\n".join(lines)

    def build_schema_entries(self) -> list[dict]:
        """构造 Embedding 召回语料：每张表（含用途）与每个字段（含表名）各一条。"""
        entries: list[dict] = []
        for t in self.table_registry_rows:
            text = f"{t['table_name']} {t.get('display_name','')} {t.get('description','')}"
            entries.append({"type": "table", "name": t["table_name"], "text": text.strip()})
        for c in self.all_columns():
            text = f"{c['table_name']} {c['column_name']} {c.get('description','')}"
            entries.append({"type": "column", "table": c["table_name"],
                            "name": c["column_name"], "text": text.strip()})
        return entries

    def match_metrics(self, question: str) -> list[MetricInfo]:
        """指标字典匹配：命中词最长的指标才注入，避免子串口径冲突。"""
        q = question.lower()
        matched: dict[str, tuple[str, MetricInfo]] = {}
        for m in self._metrics:
            names = [m["metric_name"]] + [
                s.strip() for s in (m.get("synonyms") or "").split(",") if s.strip()
            ]
            hit = ""
            for name in names:
                n = (name or "").lower()
                if n and n in q and len(n) > len(hit):
                    hit = n
            if not hit:
                continue
            matched[m["metric_name"]] = (hit, MetricInfo(
                metric_name=m["metric_name"], definition=m["definition"],
                agg_expr=m.get("agg_expr") or "", filter_expr=m.get("filter_expr") or "",
                calc_steps=m.get("calc_steps") or "", base_hint=m.get("base_hint") or "",
                unit=m.get("unit") or "", dimensions=m.get("dimensions") or "",
            ))
        # 剔除「匹配词被其他指标匹配词包含」的指标，仅保留更精确者
        cands = dict(matched)
        kept = {}
        for name, (hit, info) in cands.items():
            dominated = any(other != name and hit in other_hit
                            for other, (other_hit, _) in cands.items())
            if not dominated:
                kept[name] = (hit, info)
        return [info for _, info in kept.values()]

    async def get_watermarks(self) -> dict[str, str]:
        now = time.time()
        if now - self._watermark_ts > self.watermark_ttl or not self._watermarks:
            rows = await db.afetch_all(
                "SELECT table_name, data_watermark FROM table_registry", meta=True)
            self._watermarks = {r["table_name"]: str(r["data_watermark"] or "") for r in rows}
            self._watermark_ts = now
        return self._watermarks