#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_bm25.py — BM25 检索单元测试（含 mock 版本，无需真实 MySQL）
"""
import pytest
import time
from unittest.mock import patch, MagicMock
from services.bm25 import tokenize, BM25Retriever


class TestTokenize:
    def test_basic_chinese(self):
        tokens = tokenize("银行贷款年利率")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "贷款" in tokens or "银行贷款" in tokens

    def test_with_english(self):
        tokens = tokenize("IPO发行价格")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_empty_string(self):
        tokens = tokenize("")
        assert tokens == []


class TestBM25Retriever:
    def _make_mock_retriever(self, items=None):
        """构造一个已加载 mock 数据的 BM25Retriever。"""
        if items is None:
            items = [
                {"id": 1, "category": "banking", "question": "银行贷款年利率是多少",
                 "answer": "银行贷款年利率根据政策而定", "source": "mysql", "type": "事实性"},
                {"id": 2, "category": "corporate_finance", "question": "公司并购估值方法",
                 "answer": "常用DCF和可比公司法", "source": "mysql", "type": "推理型"},
                {"id": 3, "category": "financial_accounting", "question": "资产负债表编制方法",
                 "answer": "按会计准则编制", "source": "mysql", "type": "事实性"},
            ]
        retriever = BM25Retriever()
        retriever._docs = items
        retriever._loaded = True
        # 标记索引刚构建，避免 _load() 的自动刷新逻辑（5 分钟）触发真实 DB 重建
        retriever._last_built = time.time()
        # 手动构建 BM25 索引
        from rank_bm25 import BM25Okapi
        from services.bm25 import tokenize as _tokenize
        tokenized = [_tokenize(d["question"]) for d in items]
        retriever._bm25 = BM25Okapi(tokenized)
        return retriever

    def test_load_triggers_index(self):
        retriever = self._make_mock_retriever()
        assert retriever._loaded is True
        assert retriever._bm25 is not None
        result = retriever.search("银行贷款", top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_search_returns_list(self):
        retriever = self._make_mock_retriever()
        result = retriever.search("资产负债", top_k=5)
        assert isinstance(result, list)
        for item in result:
            assert "question" in item
            assert "score" in item
            assert "softmax_score" in item

    def test_search_with_domain_filter(self):
        retriever = self._make_mock_retriever()
        result = retriever.search("贷款", domain="banking", top_k=3)
        assert isinstance(result, list)
        for item in result:
            assert item["category"] == "banking"

    def test_get_best_match_below_threshold(self):
        retriever = self._make_mock_retriever()
        result = retriever.get_best_match("xyznotexist123", threshold=0.99)
        assert result is None

    def test_get_best_match_above_threshold(self):
        retriever = self._make_mock_retriever()
        result = retriever.get_best_match("银行贷款年利率是多少", threshold=0.01)
        assert result is not None
        assert result["category"] == "banking"
