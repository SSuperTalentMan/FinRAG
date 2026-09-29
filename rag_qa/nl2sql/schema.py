#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/schema.py — P2 内部数据结构（Pydantic DTO）。

与 ChatBI 的 app/models/dto.py 对齐，裁剪掉前端/评测用字段，保留问数链路必需的契约。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GeneratedSQL(BaseModel):
    """LLM 生成的 SQL 结构化输出（force JSON）。"""
    sql: str = ""
    chart_hint: Literal["line", "bar", "pie", "table"] | None = None
    assumptions: str = ""


class GuardResult(BaseModel):
    """SQLGuard 校验结果：通过则给出规范化 SQL 与涉及表，拒绝则给原因。"""
    ok: bool
    normalized_sql: str = ""
    tables: list[str] = []
    reject_reason: str | None = None


class MetricInfo(BaseModel):
    """指标口径卡片（零占位符）：聚合表达式 / 过滤条件 / 计算步骤 / 数据源分离。

    早期版本的残缺 SQL 片段（带 {time} 占位符）是模型照抄出错的主因；
    P2 移植时明确采用结构化卡片，避免把"有毒上下文"喂给模型。
    """
    metric_name: str
    definition: str
    agg_expr: str = ""
    filter_expr: str = ""
    calc_steps: str = ""
    base_hint: str = ""
    unit: str = ""
    dimensions: str = ""


class SchemaContext(BaseModel):
    """一次问数请求的 Schema 召回上下文包。"""
    question: str
    ddl_texts: list[str] = []
    table_names: list[str] = []
    metrics: list[MetricInfo] = []
    recall_source: str = "keyword"   # keyword / embedding / hybrid


class Nl2SqlAnswer(BaseModel):
    """问数最终答案（同步返回；同时是语义缓存存储结构）。"""
    answer_type: Literal["data", "chitchat", "clarify", "refused", "error"] = "data"
    text: str = ""
    sql: str | None = None
    assumptions: str = ""
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    tables: list[str] = []
    repair_rounds: int = 0
    latency_ms: int = 0
    status: Literal["ok", "refused", "guard_rejected", "failed", "clarify"] = "ok"
    cached: bool = False        # 是否来自语义缓存（用于观测与"数据时间"提示）