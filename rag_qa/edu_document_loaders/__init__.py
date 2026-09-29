#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/__init__.py
"""
from .edu_pdfloader import OCRPDFLoader
from .edu_docloader import OCRDOCLoader
from .edu_ocr import get_ocr

__all__ = ["OCRPDFLoader", "OCRDOCLoader", "get_ocr"]
