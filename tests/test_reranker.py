#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_reranker.py — BGE-Reranker 单元测试
"""
import pytest
from services.reranker import get_reranker, rerank


class TestReranker:
    def test_get_model_singleton(self):
        m1 = get_reranker()
        m2 = get_reranker()
        assert m1 is m2

    def test_rerank_fewer_than_topk(self):
        """候选数少于 top_k 时直接返回原序（不添加 rerank_score）。"""
        candidates = [
            {"question": "q1", "answer": "a1", "score": 0.5, "source": "bm25"},
            {"question": "q2", "answer": "a2", "score": 0.8, "source": "milvus"},
        ]
        result = rerank("测试问题", candidates, top_k=5)
        assert len(result) == 2

    def test_rerank_sorts_descending(self):
        """重排序后按 rerank_score 降序排列。"""
        candidates = [
            {"question": "q1", "answer": "a1", "score": 0.3, "source": "a"},
            {"question": "q2", "answer": "a2", "score": 0.9, "source": "b"},
            {"question": "q3", "answer": "a3", "score": 0.1, "source": "c"},
        ]
        result = rerank("公司金融问题", candidates, top_k=2)
        assert len(result) == 2
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]

    def test_rerank_returns_correct_count(self):
        candidates = [
            {"question": f"q{i}", "answer": f"a{i}", "score": 0.5, "source": "s"}
            for i in range(10)
        ]
        result = rerank("测试", candidates, top_k=3)
        assert len(result) == 3
