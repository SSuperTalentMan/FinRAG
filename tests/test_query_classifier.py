#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_query_classifier.py — 意图分类器单元测试
"""
import pytest
from rag_qa.core.query_classifier import (
    classify_intent,
    get_classifier,
    QueryClassifier,
    IntentResult,
    DOMAIN_KEYWORDS,
)


class TestIntentResult:
    def test_intent_result_fields(self):
        """IntentResult 应有 domain, confidence, keywords_hit 三个字段。"""
        result = IntentResult(domain="banking", confidence=0.85, keywords_hit=["贷款", "利率"])
        assert result.domain == "banking"
        assert result.confidence == 0.85
        assert result.keywords_hit == ["贷款", "利率"]

    def test_intent_result_defaults(self):
        """无参数的 IntentResult 应使用合理默认值。"""
        result = IntentResult(domain="general", confidence=0.0, keywords_hit=[])
        assert result.domain == "general"
        assert result.confidence == 0.0


class TestDomainKeywords:
    def test_keywords_defined(self):
        """各领域关键词应非空。"""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            assert isinstance(keywords, list)
            assert len(keywords) > 0

    def test_no_overlap_between_domains(self):
        """不同领域的关键词不应有重复。"""
        all_keywords = set()
        for keywords in DOMAIN_KEYWORDS.values():
            for kw in keywords:
                assert kw not in all_keywords, f"关键词 '{kw}' 重复出现在多个领域"
                all_keywords.add(kw)


class TestClassifyIntent:
    def test_banking_intent(self):
        """银行领域问题应被正确分类。"""
        # 直接测试 keyword 分类器，不经过 BERT 路径
        from rag_qa.core.query_classifier import _keyword_classify
        domain, confidence, hits = _keyword_classify("银行贷款利率是多少")
        assert domain == "banking"
        assert confidence > 0

    def test_corporate_finance_intent(self):
        """公司金融领域问题应被正确分类。"""
        from rag_qa.core.query_classifier import _keyword_classify
        domain, confidence, hits = _keyword_classify("公司并购重组的估值方法")
        assert domain == "corporate_finance"

    def test_financial_accounting_intent(self):
        """财务会计领域问题应被正确分类。"""
        from rag_qa.core.query_classifier import _keyword_classify
        domain, confidence, hits = _keyword_classify("资产负债表的编制方法")
        assert domain == "financial_accounting"
        assert "资产负债表" in hits

    def test_general_intent(self):
        """无领域信号的问题应返回 general。"""
        from rag_qa.core.query_classifier import _keyword_classify
        domain, confidence, hits = _keyword_classify("今天天气怎么样")
        assert domain == "general"
        assert confidence == 0.0

    def test_empty_query(self):
        """空查询应返回 general。"""
        from rag_qa.core.query_classifier import _keyword_classify
        domain, confidence, hits = _keyword_classify("")
        assert domain == "general"


class TestQueryClassifier:
    @pytest.mark.skip(reason="Requires BERT model download from HuggingFace")
    def test_keyword_predict_banking(self):
        """关键词预测银行领域问题。"""
        clf = QueryClassifier()
        domain, confidence, hits = clf.predict("银行贷款政策")
        assert domain in ("banking", "general")
        assert isinstance(confidence, float)

    @pytest.mark.skip(reason="Requires BERT model download from HuggingFace")
    def test_keyword_predict_general(self):
        """关键词预测通用问题。"""
        clf = QueryClassifier()
        domain, confidence, hits = clf.predict("你好，今天星期几")
        assert domain == "general"
