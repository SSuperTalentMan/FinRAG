#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/schemas.py — 多模态解析 / 合规审查相关 DTO（移植自 DocAudit app/models/dto.py）。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ParseUnit(BaseModel):
    """解析单元：最小粒度的版面块。"""
    page_no: int
    bbox: list[float] = [0, 0, 0, 0]
    block_type: Literal["text", "title", "table", "figure"] = "text"
    text: str
    confidence: float = 1.0
    parser_source: Literal["pymupdf", "paddleocr", "qwen_vl", "rapidocr"] = "pymupdf"


class PageResult(BaseModel):
    page_no: int
    route: Literal["digital", "ocr"]
    units: list[ParseUnit] = []
    text: str = ""
    confidence: float = 1.0
    parser_source: str = "pymupdf"
    error: str | None = None


class ClauseOut(BaseModel):
    clause_no: str
    title: str = ""
    content: str
    page_start: int
    page_end: int
    extract_status: Literal["ok", "partial", "failed"] = "ok"
    defects: str = ""


class ViolatedRule(BaseModel):
    rule_id: str
    doc_name: str
    article_no: str = ""
    evidence_quote: str = ""


class CompareOutModel(BaseModel):
    """比对 Agent 输出 Schema（结构化输出校验用）。"""
    verdict: Literal["compliant", "violation", "insufficient_evidence"] = "insufficient_evidence"
    violated_rules: list[dict] = []
    evidence: str = ""
    suggestion: str = ""


class GradeOutModel(BaseModel):
    """分级 Agent 输出 Schema（risk_level 合法性由调用方校验并兜底）。"""
    risk_level: str = ""
    needs_human: bool = False


class ClauseReviewOut(BaseModel):
    clause_no: str
    clause_id: int = 0
    verdict: Literal["compliant", "violation", "insufficient_evidence"]
    risk_level: Literal["high", "medium", "low"]
    violated_rules: list[ViolatedRule] = []
    evidence: str = ""
    suggestion: str = ""
    hitl_status: str | None = None