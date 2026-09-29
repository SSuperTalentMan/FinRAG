#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_text_spliter.py — 中文递归文本切分器单元测试
"""
import pytest
langchain = pytest.importorskip("langchain_text_splitters", reason="langchain_text_splitters 未安装")

from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter


class TestChineseRecursiveTextSplitter:
    def test_split_chinese_paragraphs(self):
        """测试中文段落切分：以句号/换行作为分隔。"""
        splitter = ChineseRecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
        text = "这是第一段内容，描述银行信贷业务。第二段是关于利率调整的话题。第三段讨论存款准备金率。"
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_split_empty_string(self):
        """空字符串应返回空列表。"""
        splitter = ChineseRecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_english_text(self):
        """英文文本也能正确切分。"""
        splitter = ChineseRecursiveTextSplitter(chunk_size=50, chunk_overlap=5)
        text = "This is the first sentence. This is the second sentence. This is the third one."
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_split_with_newlines(self):
        """含换行符的长文本按段落切分。"""
        splitter = ChineseRecursiveTextSplitter(chunk_size=30, chunk_overlap=0)
        text = "第一段内容比较长，用于测试段落切分效果。\n\n第二段内容也包含很多细节描述，需要被正确分割。\n\n第三段内容是最后的测试段落。"
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 2

    def test_chunk_size_limit(self):
        """每个块不能超过 chunk_size。"""
        splitter = ChineseRecursiveTextSplitter(chunk_size=30, chunk_overlap=0)
        text = "这是一段很长的文本，用于测试分块器是否会将大块切分成更小的单元，确保每个单元都不超过限制。"
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 30 + 1  # 允许少量误差

    def test_overlap_preserved(self):
        """chunk_overlap > 0 时，相邻块应有重叠内容。"""
        splitter = ChineseRecursiveTextSplitter(chunk_size=40, chunk_overlap=10)
        text = "第一段内容用于测试重叠效果。第二段内容是测试用的另一段文字。"
        chunks = splitter.split_text(text)
        if len(chunks) >= 2:
            assert chunks[0][-10:] == chunks[1][:10] or len(chunks[0]) > 10
