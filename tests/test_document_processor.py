#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_document_processor.py — 文档处理器单元测试
"""
import pytest
langchain_community = pytest.importorskip("langchain_community", reason="langchain_community 未安装")
pymupdf = pytest.importorskip("pymupdf", reason="pymupdf (fitz) 未安装")

from rag_qa.core.document_processor import load_documents_from_directory, DOCUMENT_LOADERS


class TestDocumentLoaders:
    def test_loader_mapping(self):
        """验证扩展名到加载器的映射包含所有期望类型。"""
        assert ".txt" in DOCUMENT_LOADERS
        assert ".pdf" in DOCUMENT_LOADERS
        assert ".docx" in DOCUMENT_LOADERS
        assert ".md" in DOCUMENT_LOADERS

    def test_loader_returns_callable(self):
        """每个加载器应返回可调用对象。"""
        for ext, loader_fn in DOCUMENT_LOADERS.items():
            assert callable(loader_fn)


class TestLoadDocumentsFromDirectory:
    def test_nonexistent_directory(self, tmp_path):
        """不存在的目录应返回空列表。"""
        result = load_documents_from_directory(str(tmp_path / "nonexistent_dir"))
        assert result == []

    def test_empty_directory(self, tmp_path):
        """空目录应返回空列表。"""
        result = load_documents_from_directory(str(tmp_path))
        assert result == []

    def test_txt_file_loaded(self, tmp_path):
        """测试 .txt 文件能被正确加载。"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("这是一段测试文本，用于验证文档加载功能。", encoding="utf-8")
        result = load_documents_from_directory(str(tmp_path))
        assert len(result) == 1
        assert "测试文本" in result[0].page_content
        assert result[0].metadata["source"] == "tmp_path"

    def test_unsupported_extension_skipped(self, tmp_path):
        """不支持的文件类型应被跳过。"""
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3")
        result = load_documents_from_directory(str(tmp_path))
        assert result == []
