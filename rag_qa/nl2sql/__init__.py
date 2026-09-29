#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql — 结构化数据问数（融合 ChatBI, P2）。

把「自然语言 → Schema 召回 → SQL 生成 → SQLGuard 纵深防御 → 只读执行」链路收口进 FinRag，
与 P1 的非结构化文档 RAG 构成"双引擎"。P3 将由 LangGraph AnswerGraph 统一路由到二者。
"""