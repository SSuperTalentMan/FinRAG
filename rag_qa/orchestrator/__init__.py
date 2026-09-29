#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/orchestrator — LangGraph 统一编排层（融合 ChatBI/DocAudit, P3）。

把四种输入路由到对应 Skill / Workflow：
- 非结构化文档问答  -> RAG Skill（复用 FinRag 混合检索）
- 结构化问数        -> NL2SQL Skill（融合 ChatBI，SQLGuard 纵深防御）
- 读图/扫描件问题   -> 多模态解析 + RAG
- 寒暄              -> 通用回答
"""

from __future__ import annotations

from rag_qa.orchestrator.state import AnswerState
from rag_qa.orchestrator.route import SKILLS, _decide_skill, decide_skill_keywords

__all__ = [
    "AnswerState",
    "SKILLS",
    "_decide_skill",
    "decide_skill_keywords",
]