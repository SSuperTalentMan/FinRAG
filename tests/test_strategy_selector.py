#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_strategy_selector.py — 策略选择器单元测试（mock LLM，无需真实 API）
"""
import pytest
from unittest.mock import patch
from rag_qa.core.strategy_selector import (
    StrategySelector,
    STRATEGY_DIRECT,
    STRATEGY_HYDE,
    STRATEGY_SUBQUERY,
    STRATEGY_BACKTRACK,
)


class TestStrategySelector:
    def test_all_strategies_defined(self):
        """所有策略常量应正确定义。"""
        from rag_qa.core.strategy_selector import ALL_STRATEGIES
        assert STRATEGY_DIRECT in ALL_STRATEGIES
        assert STRATEGY_HYDE in ALL_STRATEGIES
        assert STRATEGY_SUBQUERY in ALL_STRATEGIES
        assert STRATEGY_BACKTRACK in ALL_STRATEGIES
        assert len(ALL_STRATEGIES) == 4

    @patch("services.llm.chat_completion")
    def test_select_strategy_returns_valid(self, mock_chat):
        """select_strategy 应返回有效策略。"""
        mock_chat.return_value = "直接检索"
        selector = StrategySelector()
        result = selector.select_strategy("银行贷款是什么")
        from rag_qa.core.strategy_selector import ALL_STRATEGIES
        assert result in ALL_STRATEGIES

    @patch("services.llm.chat_completion")
    def test_select_strategy_hyde(self, mock_chat):
        """HyDE 策略选择应正常。"""
        mock_chat.return_value = "假设问题检索，因为问题较抽象"
        selector = StrategySelector()
        result = selector.select_strategy("人工智能对未来经济的影响")
        assert result == STRATEGY_HYDE

    @patch("services.llm.chat_completion", side_effect=Exception("API quota exceeded"))
    def test_select_strategy_fallback_on_error(self, mock_chat):
        """策略选择失败时应返回 STRATEGY_DIRECT。"""
        selector = StrategySelector()
        result = selector.select_strategy("你好")
        assert result == STRATEGY_DIRECT
