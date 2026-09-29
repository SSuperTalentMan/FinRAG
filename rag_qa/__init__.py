#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/__init__.py
"""
from .core import RAGPrompts, StrategySelector, QueryClassifier, classify_intent, IntentResult

__all__ = [
    "RAGPrompts",
    "StrategySelector",
    "QueryClassifier",
    "classify_intent",
    "IntentResult",
    "RAGSystem",
]


def __getattr__(name: str):
    """RAGSystem 延迟到首次访问时才加载（依赖重 ML 库，见 rag_qa/core/__init__.py）。"""
    if name == "RAGSystem":
        from .core import RAGSystem
        globals()["RAGSystem"] = RAGSystem
        return RAGSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
