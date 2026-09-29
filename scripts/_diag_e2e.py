#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端实证：用修复后的分类器跑真实 chat 管线（classify -> _build_context -> LLM），
验证 doc_4/doc_31 内容能否真正进入最终回答。无需重启服务。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="ERROR")

from rag_qa.core.query_classifier import classify_intent
from routers.chat import _build_context
from services.llm import generate_answer

QUERIES = [
    "注册制下发行人信息披露有哪些要求？",
    "股票发行注册制改革的主要内容是什么？",
    "投资者教育包括哪些内容？",
]

for q in QUERIES:
    intent = classify_intent(q)
    context, sources = _build_context(q, intent)
    print("=" * 70)
    print(f"Q: {q}")
    print(f"  分类域={intent.domain}  召回源数={len(sources)}")
    for s in sources[:5]:
        print(f"    - {s['question'][:50]}  score={s['score']}")
    try:
        ans = generate_answer(q, context, intent.domain)
    except Exception as e:
        ans = f"[LLM 调用失败: {e}]"
    print(f"  【LLM 回答前 320 字】\n  {ans[:320]}")
