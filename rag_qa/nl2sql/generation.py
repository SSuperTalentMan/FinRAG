#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/generation.py — SQL 生成（复用 FinRag services/llm_ext 结构化调用）。

- 上下文由 prompts 层渲染（DDL + 指标口径卡片 + 失败回馈 + 隔离问句）；
- 走 chat_json_validated：Pydantic 校验失败回喂重试一次；
- 模型默认取 config.llm.sql_model（留空则回落到 model，即 qwen3.7-flash），qwen3 自动关思考。
"""
from __future__ import annotations

import logging

from config import get_config
from rag_qa.nl2sql import prompts
from rag_qa.nl2sql.schema import GeneratedSQL, SchemaContext
from services.llm_ext import chat_json_validated

logger = logging.getLogger(__name__)

VALID_HINTS = {"line", "bar", "pie", "table"}


def build_sql_messages(ctx: SchemaContext, failure_context: str = "") -> list[dict]:
    sections: list[str] = []
    sections += prompts.render_ddl(ctx.ddl_texts)
    sections += prompts.render_metric_cards(ctx.metrics)
    sections += prompts.render_fewshot()
    sections.append("【当前问题(见下方标签内文本)】")
    sections.append(prompts.wrap_question(ctx.question))
    sections.append("")
    sections += prompts.render_failure(failure_context)
    return [
        {"role": "system", "content": prompts.SQL_SYSTEM},
        {"role": "user", "content": "\n".join(sections).strip()},
    ]


async def generate_sql(ctx: SchemaContext, failure_context: str = "") -> tuple[GeneratedSQL, dict, str]:
    """生成 SQL。返回 (GeneratedSQL, usage, 拿到的原始 content)。

    空 sql 且 assumptions 以 REFUSE 开头 → 表示拒答（data unavailable）。
    """
    from config import get_config
    cfg = get_config()
    model = cfg.llm.sql_model or cfg.llm.model
    messages = build_sql_messages(ctx, failure_context)
    made, usage = await chat_json_validated(
        messages, model=model, schema=GeneratedSQL, temperature=0.1, stage="nl2sql",
        max_tokens=cfg.nl2sql.gen_max_tokens,
    )
    return made, usage, made.sql