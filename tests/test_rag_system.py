#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_rag_system.py — RAG 系统单元测试（含 mock 版本，无需真实 Milvus）
"""
import pytest
from unittest.mock import patch, MagicMock
from rag_qa.core.rag_system import RAGSystem, STRATEGY_DIRECT, STRATEGY_HYDE, STRATEGY_SUBQUERY, STRATEGY_BACKTRACK


class TestRAGSystem:
    def test_init(self):
        """RAGSystem 初始化应正常。"""
        system = RAGSystem()
        assert system is not None
        assert system.strategy_selector is not None

    def test_strategy_constants(self):
        """验证策略常量定义正确。"""
        assert STRATEGY_DIRECT == "直接检索"
        assert STRATEGY_HYDE == "假设问题检索"
        assert STRATEGY_SUBQUERY == "子查询检索"
        assert STRATEGY_BACKTRACK == "回溯问题检索"

    @patch("rag_qa.core.rag_system.encode_query_dense_sparse")
    @patch("rag_qa.core.rag_system.get_milvus_client")
    @patch("rag_qa.core.rag_system.search_milvus")
    @patch("rag_qa.core.rag_system.get_bm25_retriever")
    def test_retrieve_direct_no_crash(self, mock_bm25, mock_search, mock_client, mock_encode):
        """直接检索不应抛出异常（数据为空时返回空列表）。"""
        mock_encode.return_value = ([0.0] * 1024, MagicMock())
        # has_collection=False 使 ensure_collection 走新建分支（避免 schema 检查需要真实 fields）
        mock_client.return_value = MagicMock()
        mock_client.return_value.has_collection.return_value = False
        mock_search.return_value = []
        mock_bm25.return_value.search.return_value = []

        system = RAGSystem()
        result = system.retrieve("测试问题", strategy=STRATEGY_DIRECT, k=3)
        assert isinstance(result, list)
        assert result == []

    @patch("rag_qa.core.rag_system.encode_query_dense_sparse")
    @patch("rag_qa.core.rag_system.get_milvus_client")
    @patch("rag_qa.core.rag_system.search_milvus")
    @patch("rag_qa.core.rag_system.get_bm25_retriever")
    def test_retrieve_returns_candidates(self, mock_bm25, mock_search, mock_client, mock_encode):
        """检索应返回候选列表。"""
        mock_encode.return_value = ([0.0] * 1024, MagicMock())
        # has_collection=False 使 ensure_collection 走新建分支（避免 schema 检查需要真实 fields）
        mock_client.return_value = MagicMock()
        mock_client.return_value.has_collection.return_value = False
        mock_search.return_value = [
            {"score": 0.9, "question": "q1", "answer": "a1", "category": "banking", "source": "milvus_dense"},
        ]
        mock_bm25.return_value.search.return_value = []

        system = RAGSystem()
        result = system.retrieve("测试问题", strategy=STRATEGY_DIRECT, k=3)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["question"] == "q1"

    @patch("rag_qa.core.rag_system.encode_query_dense_sparse")
    @patch("rag_qa.core.rag_system.get_milvus_client")
    @patch("rag_qa.core.rag_system.search_milvus")
    @patch("rag_qa.core.rag_system.get_bm25_retriever")
    def test_build_context_returns_tuple(self, mock_bm25, mock_search, mock_client, mock_encode):
        """build_context 应返回 (context_str, sources_list)。"""
        mock_encode.return_value = ([0.0] * 1024, MagicMock())
        # has_collection=False 使 ensure_collection 走新建分支（避免 schema 检查需要真实 fields）
        mock_client.return_value = MagicMock()
        mock_client.return_value.has_collection.return_value = False
        mock_search.return_value = []
        mock_bm25.return_value.search.return_value = []

        system = RAGSystem()
        context, sources = system.build_context("测试", top_k=3)
        assert isinstance(context, str)
        assert isinstance(sources, list)
