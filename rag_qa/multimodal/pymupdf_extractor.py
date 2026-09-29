#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""langgraph/multimodal/pymupdf_extractor.py — PyMuPDF：文本层检测 / 文本块提取（带 bbox）/ 页面渲染图。"""
from __future__ import annotations

import fitz

from rag_qa.review.schemas import ParseUnit


def open_doc(file_path: str) -> fitz.Document:
    return fitz.open(file_path)


def page_text_chars(page: fitz.Page) -> int:
    return len(page.get_text("text").strip())


def extract_units(page: fitz.Page, page_no: int) -> list[ParseUnit]:
    """按版面块提取：字号 ≥ 正文 1.15 倍或加粗判定为标题。"""
    units: list[ParseUnit] = []
    d = page.get_text("dict")
    body_sizes = [
        s["size"] for b in d["blocks"] if b.get("type") == 0
        for l in b.get("lines", []) for s in l.get("spans", [])
    ]
    body_size = sorted(body_sizes)[len(body_sizes) // 2] if body_sizes else 11.0
    for b in d.get("blocks", []):
        if b.get("type") == 1:  # 图片块
            units.append(ParseUnit(
                page_no=page_no, bbox=[round(x, 1) for x in b["bbox"]],
                block_type="figure", text="[图片]", confidence=0.5, parser_source="pymupdf",
            ))
            continue
        lines: list[str] = []
        max_size = 0.0
        for l in b.get("lines", []):
            line_text = "".join(s["text"] for s in l.get("spans", []))
            if not line_text.strip():
                continue
            lines.append(line_text)
            max_size = max(max_size, max(s["size"] for s in l.get("spans", [])))
        text = "\n".join(lines).strip()
        if not text:
            continue
        is_title = max_size >= body_size * 1.15 and len(text) < 60
        units.append(ParseUnit(
            page_no=page_no, bbox=[round(x, 1) for x in b["bbox"]],
            block_type="title" if is_title else "text", text=text,
            confidence=1.0, parser_source="pymupdf",
        ))
    return units


def render_page_png(page: fitz.Page, dpi: int = 110) -> bytes:
    return page.get_pixmap(dpi=dpi).tobytes("png")