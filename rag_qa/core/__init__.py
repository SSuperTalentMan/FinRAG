#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/core/__init__.py
"""
from .prompts import RAGPrompts
from .strategy_selector import StrategySelector, STRATEGY_DIRECT, STRATEGY_HYDE, STRATEGY_SUBQUERY, STRATEGY_BACKTRACK
from .query_classifier import QueryClassifier, classify_intent, get_classifier, IntentResult

__all__ = [
    "RAGPrompts",
    "StrategySelector",
    "STRATEGY_DIRECT",
    "STRATEGY_HYDE",
    "STRATEGY_SUBQUERY",
    "STRATEGY_BACKTRACK",
    "QueryClassifier",
    "classify_intent",
    "get_classifier",
    "IntentResult",
    "RAGSystem",
]


def __getattr__(name: str):
    """懒加载 RAGSystem：它依赖 services.embedding（FlagEmbedding 等重 ML 依赖），
    若在包导入时急切加载，会让任何无关模块（如仅需 QueryClassifier 的工具脚本）
    也被迫加载 torch/FlagEmbedding，拖慢启动、放大依赖缺失风险。"""
    if name == "RAGSystem":
        from .rag_system import RAGSystem
        globals()["RAGSystem"] = RAGSystem
        return RAGSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
