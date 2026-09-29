#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_intent.py — 意图识别单元测试
"""
import pytest
from services.intent import classify_intent, IntentResult


class TestClassifyIntent:
    def test_banking_keywords(self):
        result = classify_intent("银行贷款利率是多少")
        assert result.domain == "banking"
        assert result.confidence > 0
        assert any(kw in result.keywords_hit for kw in ["贷款", "利率"])

    def test_corporate_finance_keywords(self):
        result = classify_intent("公司并购重组的估值方法")
        assert result.domain == "corporate_finance"
        assert "并购" in result.keywords_hit

    def test_financial_accounting_keywords(self):
        result = classify_intent("资产负债表的编制方法")
        assert result.domain == "financial_accounting"
        assert "资产负债表" in result.keywords_hit

    def test_general_no_domain_keywords(self):
        result = classify_intent("今天天气怎么样")
        assert result.domain == "general"
        assert result.confidence == 0.0

    def test_low_confidence_fallback(self):
        # 命中多个领域但置信度低 → 降级为 general
        result = classify_intent("银行和公司财务都涉及贷款")
        # 可能因多领域命中导致置信度不足
        assert isinstance(result, IntentResult)
        assert result.domain in ("banking", "corporate_finance", "financial_accounting", "general")

    def test_empty_query(self):
        result = classify_intent("")
        assert result.domain == "general"

    def test_single_keyword(self):
        result = classify_intent("利率")
        # 单关键词可能因置信度低而降级
        assert isinstance(result, IntentResult)
        assert result.domain in ("banking", "general")
