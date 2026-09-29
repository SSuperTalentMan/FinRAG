#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/e2e_verify_qa.py — FinRag 端到端真实问答验证（含来源标注核查）

驱动生产管线：意图分类 → 检索(_build_context, 已回填真实来源) → LLM 生成回答。
对一组政策/金融问题打印：意图、检索来源(真实 source + url)、最终答案，并检测
qwen3 是否泄漏思考链。

用法：.venv/Scripts/python.exe scripts/e2e_verify_qa.py
"""
import os
import sys
import time
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag_qa.core.query_classifier import classify_intent
import routers.chat as chat_router
from services.llm import generate_answer

QUESTIONS = [
    "货币政策如何支持实体经济",
    "个人养老金怎么参加",
    "科创板注册制改革方向",
    "上市公司分红有哪些规定",
    "地方政府债务风险如何防范",
    "什么是普惠金融",
]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str):
    """剥离 qwen3 思考链，返回 (clean, had_think)"""
    if "<think>" in text:
        return _THINK.sub("", text).strip(), True
    return text.strip(), False


def main():
    print("#" * 72)
    print("# FinRag 端到端真实问答验证（检索 + 来源标注 + LLM 生成）")
    print("#" * 72)
    for idx, q in enumerate(QUESTIONS, 1):
        print("\n" + "=" * 72)
        print(f"[#{idx}] 问题：{q}")
        t0 = time.time()
        # Step 1: 意图分类
        try:
            intent = classify_intent(q)
        except Exception as e:
            print("  意图分类失败:", repr(e))
            continue
        print(f"  意图 → domain={intent.domain}  confidence={intent.confidence:.4f}  hits={intent.keywords_hit}")
        # Step 2: 检索 + 来源回填（生产路径）
        try:
            context, sources = chat_router._build_context(q, intent)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  检索失败:", repr(e))
            continue
        print(f"  检索 → 命中 {len(sources)} 条（耗时 {time.time()-t0:.1f}s）")
        for i, s in enumerate(sources, 1):
            src = s.get("source") or "(空)"
            url = (s.get("source_url") or "")
            url_disp = (url[:72] + "…") if len(url) > 72 else url
            print(f"    {i}. score={s.get('score'):.4f}")
            print(f"       来源：{src}")
            if url_disp:
                print(f"       链接：{url_disp}")
            print(f"       命中问：{s.get('question', '')[:60]}")
        # Step 3: LLM 生成
        if not context:
            print("  无检索上下文 → LLM 直接作答")
            continue
        try:
            t1 = time.time()
            raw = generate_answer(q, context, intent.domain)
            answer, had_think = strip_think(raw)
            print(f"  LLM 作答（耗时 {time.time()-t1:.1f}s，思考链={'是' if had_think else '否'}）：")
            print("  " + "\n  ".join(answer.split("\n")))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  LLM 生成失败:", repr(e))
    print("\n" + "=" * 72)
    print("验证结束。")


if __name__ == "__main__":
    main()
