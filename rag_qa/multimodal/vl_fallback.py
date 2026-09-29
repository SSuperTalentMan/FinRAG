#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/multimodal/vl_fallback.py — 低置信扫描页的 qwen-vl 视觉兜底转写。"""
from __future__ import annotations

import logging

from rag_qa.review.schemas import ParseUnit
from rag_qa.review.prompts import VL_PAGE_SYSTEM
from services.llm_ext import chat_vl

logger = logging.getLogger(__name__)


async def transcribe_page(page_png: bytes, page_no: int) -> list[ParseUnit]:
    try:
        data, usage = await chat_vl(VL_PAGE_SYSTEM, page_png)
    except Exception as e:  # noqa: BLE001
        # 视觉模型链全部不可用（如额度耗尽）：显式标记空页并告警，绝不静默降级到纯文本模型
        logger.warning("vl_fallback page=%s 视觉模型全部不可用，该页标记为空: %s", page_no, str(e)[:160])
        return [ParseUnit(page_no=page_no, text="", confidence=0.0, parser_source="qwen_vl")]
    logger.info("vl_fallback page=%s tokens_in=%s", page_no, usage.get("tokens_in", 0))
    units: list[ParseUnit] = []
    for blk in data.get("blocks", []):
        text = (blk.get("text") or "").strip()
        if not text:
            continue
        units.append(ParseUnit(
            page_no=page_no, bbox=[0, 0, 0, 0],
            block_type=blk.get("type", "text") if blk.get("type") in ("title", "text", "table") else "text",
            text=text, confidence=0.99, parser_source="qwen_vl",
        ))
    if not units:
        units.append(ParseUnit(page_no=page_no, text="", confidence=0.0, parser_source="qwen_vl"))
    return units