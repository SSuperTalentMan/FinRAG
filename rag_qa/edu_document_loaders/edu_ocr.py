#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/edu_ocr.py

RapidOCR 引擎封装，优先使用 Paddle 引擎（GPU加速），回退 ONNXRuntime（CPU友好）。
"""


def get_ocr(use_cuda: bool = True):
    """
    获取 RapidOCR 实例。
    有 GPU 时优先使用 rapidocr_paddle，否则使用 rapidocr_onnxruntime。
    """
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr


def ocr_image(image_path: str) -> list[tuple[str, tuple]]:
    """
    对单张图片执行 OCR，返回 [(文本, bbox), ...]。
    """
    ocr = get_ocr()
    result, _ = ocr(image_path)
    return result if result else []
