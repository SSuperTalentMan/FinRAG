#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/multimodal/ocr_parser.py — 扫描页 OCR 引擎适配层。

生产引擎：RapidOCR（ONNXRuntime，PP-OCRv4 转换模型，自包含、Windows CPU 稳定）。
设计依据：PaddleOCR 3.x 在 Windows CPU 上反复踩中文路径 Config 失败、PIR+oneDNN
NotImplementedError、静默空结果三类坑，最终以 RapidOCR 作为生产引擎、
复杂版面页由 qwen-vl-max 兜底。环境变量 FINRAG_OCR_ENGINE=ppstructure 可切回。
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from pathlib import Path

# paddle C++ 层对含非 ASCII 字符的 Windows 用户目录检查会失败，须在导入前把模型缓存指到 ASCII 路径
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", r"C:\paddlex_cache")

from rag_qa.review.schemas import ParseUnit

logger = logging.getLogger(__name__)

RE_CLAUSE_TITLE = re.compile(r"^第[一二三四五六七八九十百\d]+条")

_engine = None
_engine_name: str | None = None
_lock = threading.Lock()
_ocr_call_lock = threading.Lock()  # OCR 引擎实例非线程安全，调用需串行化


def _get_engine():
    """懒加载 OCR 引擎（进程内单例，默认 rapidocr）。"""
    global _engine, _engine_name
    if _engine is not None:
        return _engine, _engine_name
    with _lock:
        if _engine is not None:
            return _engine, _engine_name
        choice = os.getenv("FINRAG_OCR_ENGINE", "rapidocr")
        if choice == "ppstructure":
            try:
                from paddleocr import PPStructureV3
                logger.info("初始化 PPStructureV3（CPU，首次运行自动下载模型）...")
                eng = PPStructureV3(
                    device="cpu", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False,
                )
                _engine, _engine_name = eng, "ppstructure"
                return _engine, _engine_name
            except Exception as e:  # noqa: BLE001
                logger.warning("PPStructureV3 不可用（%s），降级 RapidOCR", str(e)[:120])
        from rapidocr_onnxruntime import RapidOCR
        logger.info("初始化 RapidOCR（ONNXRuntime CPU）...")
        _engine, _engine_name = RapidOCR(), "rapidocr"
        return _engine, _engine_name


def _looks_like_title(line: str) -> bool:
    return bool(RE_CLAUSE_TITLE.search(line)) or (len(line) <= 30 and not re.search(r"[。;;,,]$", line) and len(line) > 3)


def _parse_rapidocr(image_path: str) -> list[dict]:
    engine, _ = _get_engine()
    raw, _elapse = engine(image_path)
    blocks: list[dict] = []
    for item in raw or []:
        # RapidOCR 1.x: [box, text, score]; 2.x: 对象带 .box/.txt/.score
        if isinstance(item, dict):
            text, score = str(item.get("txt") or item.get("text") or ""), float(item.get("score") or 0)
            box = item.get("box") or item.get("bbox") or [0, 0, 0, 0]
        else:
            box, text, score = item[0], str(item[1]), float(item[2]) if len(item) > 2 else 0.9
        if not text.strip():
            continue
        blocks.append({
            "type": "title" if _looks_like_title(text) else "text",
            "text": text.strip(),
            "bbox": [float(x) for p in box for x in p][:4] if box else [0, 0, 0, 0],
            "score": score,
        })
    return blocks


def _parse_sync(image_path: str) -> list[dict]:
    engine, name = _get_engine()
    with _ocr_call_lock:
        blocks = _parse_rapidocr(image_path)
    return blocks


def _avg_score(blocks: list[dict]) -> float:
    if not blocks:
        # OCR 零结果（静默空输出）必须视为不可信，交给置信度路由升级 qwen-vl 兜底
        return 0.0
    scores = [b.get("score") for b in blocks if b.get("score") is not None]
    return sum(scores) / len(scores) if scores else 0.9


async def parse_page(image_path: str, page_no: int) -> tuple[list[ParseUnit], float]:
    """解析一页扫描图，返回 (units, 平均置信度)。"""
    blocks = await asyncio.to_thread(_parse_sync, str(image_path))
    conf = _avg_score(blocks)
    _, engine_name = _get_engine()
    units = [
        ParseUnit(
            page_no=page_no,
            bbox=[round(float(x), 1) for x in (b.get("bbox") or [0, 0, 0, 0])][:4],
            block_type=b.get("type", "text"),
            text=b["text"],
            confidence=round(float(b.get("score", conf)), 4),
            parser_source=engine_name if engine_name in ("rapidocr", "paddleocr") else "paddleocr",
        )
        for b in blocks
    ]
    return units, conf


def save_page_image(page_png: bytes, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_bytes(page_png)
    return p