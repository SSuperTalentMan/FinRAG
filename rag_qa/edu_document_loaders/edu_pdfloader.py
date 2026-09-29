#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/edu_document_loaders/edu_pdfloader.py

基于 pymupdf + RapidOCR 的 PDF 文档加载器。
支持纯文本 PDF 和图片 PDF（OCR）。
"""

import os
import tempfile
from pathlib import Path
from typing import Iterator
import fitz  # pymupdf
from langchain_core.documents import Document
from .edu_ocr import get_ocr
from loguru import logger


class OCRPDFLoader:
    """PDF 文档加载器：文本页直接提取，图片页使用 OCR。"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {self.file_path}")

    def load(self) -> list[Document]:
        docs = []
        ocr = get_ocr()
        doc = fitz.open(str(self.file_path))

        for page_num in range(len(doc)):
            page = doc[page_num]
            # 尝试提取文本
            text = page.get_text().strip()
            if text:
                # 有文本，直接使用
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(self.file_path),
                        "page": page_num + 1,
                        "type": "text",
                    },
                ))
            else:
                # 纯图片页，使用 OCR
                try:
                    pix = page.get_pixmap(dpi=200)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                    pix.save(tmp_path)
                    result, _ = ocr(tmp_path)
                    os.unlink(tmp_path)
                    if result:
                        ocr_text = "\n".join([line[0] for line in result])
                        if ocr_text.strip():
                            docs.append(Document(
                                page_content=ocr_text,
                                metadata={
                                    "source": str(self.file_path),
                                    "page": page_num + 1,
                                    "type": "ocr",
                                },
                            ))
                except Exception as e:
                    logger.warning(f"PDF 第 {page_num + 1} 页 OCR 失败: {e}")

        doc.close()
        logger.info(f"PDF 加载完成: {self.file_path}, 共 {len(docs)} 页")
        return docs
