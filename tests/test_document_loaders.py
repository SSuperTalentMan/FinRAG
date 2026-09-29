#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_document_loaders.py — 文档加载器单元测试
"""
import pytest
pymupdf = pytest.importorskip("pymupdf", reason="pymupdf (fitz) 未安装")
docx = pytest.importorskip("python-docx", reason="python-docx 未安装")

from pathlib import Path
from rag_qa.edu_document_loaders import get_ocr, ocr_image


class TestOCREngine:
    def test_get_ocr_returns_instance(self):
        """get_ocr() 应返回一个 RapidOCR 实例。"""
        ocr = get_ocr()
        assert ocr is not None

    def test_ocr_image_with_nonexistent(self):
        """传入不存在的路径应返回空列表。"""
        result = ocr_image("/nonexistent/path/image.png")
        assert result == []


class TestOCRModule:
    def test_import_all_exports(self):
        """验证所有导出项都可导入。"""
        from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader
        assert OCRPDFLoader is not None
        assert OCRDOCLoader is not None

    def test_ocrpdfloader_init_fails_on_missing_file(self):
        """初始化时文件不存在应抛出 FileNotFoundError。"""
        from rag_qa.edu_document_loaders import OCRPDFLoader
        with pytest.raises(FileNotFoundError):
            OCRPDFLoader("/nonexistent/file.pdf")

    def test_ocrdocloader_init_fails_on_missing_file(self):
        """初始化时文件不存在应抛出 FileNotFoundError。"""
        from rag_qa.edu_document_loaders import OCRDOCLoader
        with pytest.raises(FileNotFoundError):
            OCRDOCLoader("/nonexistent/file.docx")
