#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/multimodal/parsing_router.py — 页级解析路由：digital / ocr / qwen_vl。"""
from __future__ import annotations

from config import get_config


def route_page(text_layer_chars: int) -> str:
    """按文本层字符数判定：≥ 阈值 → digital（PyMuPDF 直提）；否则 → ocr。"""
    return "digital" if text_layer_chars >= get_config().multimodal.digital_min_chars else "ocr"


def route_ocr_confidence(avg_confidence: float) -> bool:
    """OCR 结果置信度二次路由：低置信 → 需 qwen-vl 兜底。"""
    return avg_confidence < get_config().multimodal.ocr_confidence_route