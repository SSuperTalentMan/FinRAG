#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_orchestrator.py — LangGraph 统一编排层（P3）单元测试。

覆盖范围（全部离线、不需 MySQL/Redis/Milvus/真实 LLM）：
  1. 路由规则 `_decide_skill` 的关键样例（四类 Skill）
  2. 路由准确率门禁：在 scripts/route_skill_eval.json（218 题）上 ≥ 90%
  3. `decide_skill_keywords` 诊断输出一致性
  4. `new_state` 状态默认值
  5. AnswerGraph 构建与条件边分支逻辑（multimodal 归并 rag）
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_qa.orchestrator.route import _decide_skill, decide_skill_keywords
from rag_qa.orchestrator.state import new_state

DS_PATH = Path(__file__).resolve().parent.parent / "scripts" / "route_skill_eval.json"


@pytest.mark.parametrize(
    "question, want",
    [
        ("华东区近7天销售额是多少?", "nl2sql"),
        ("2025年各门店GMV排名", "nl2sql"),
        ("退货率最高的三个商品是什么", "nl2sql"),
        ("你好", "chitchat"),
        ("请问你都会什么?", "chitchat"),
        ("识别这张发票图片上的金额", "multimodal"),
        ("扫描一下合同扫描件的有效期", "multimodal"),
        ("存款保险的偿付限额是多少", "rag"),
        ("股票发行注册制改革的主要内容", "rag"),
        ("一款模型平均每单毛利是多少", "nl2sql"),
    ],
)
def test_decide_skill_typicals(question, want):
    assert _decide_skill(question) == want


def test_route_accuracy_gate():
    """路由准确率门禁：在标注数据集上 ≥ 90%（离线、确定性、可复现）。"""
    data = json.loads(DS_PATH.read_text(encoding="utf-8"))
    assert len(data) >= 200, f"评测数据集应≥200题，当前 {len(data)}"
    correct = sum(1 for it in data if _decide_skill(it["question"]) == it["skill"])
    acc = correct / len(data)
    assert acc >= 0.90, f"路由准确率 {acc:.2%} 低于门禁 90%"


def test_decide_skill_keywords_consistency():
    skill, kata = decide_skill_keywords("华东区近7天销售额")
    assert skill == "nl2sql"
    assert "销售额" in kata
    # 无信号问题：落到 rag 且无命中词
    skill2, kata2 = decide_skill_keywords("一般人理财怎么入门")
    assert skill2 == "rag"
    assert kata2 == []


def test_new_state_defaults():
    s = new_state("你好", role="admin", user_id=7)
    assert s["question"] == "你好"
    assert s["role"] == "admin"
    assert s["user_id"] == 7
    assert s["skill"] == "rag"        # 尚未路由
    assert s["status"] == "ok"
    assert s["degraded"] is False
    assert s["sources"] == []
    assert s["trace"] == ["init"]


def test_answer_graph_build_and_branch():
    """AnswerGraph 能编译，且条件边分支符合预期（multimodal 归并到 rag 检索）。"""
    from rag_qa.orchestrator import graph as gm

    g = gm.build_answer_graph()
    assert hasattr(g, "ainvoke") and hasattr(g, "astream")

    gr = g.get_graph()
    names = set(gr.nodes.keys()) if hasattr(gr.nodes, "keys") else {n.name for n in gr.nodes}
    assert {"route", "rag_retrieve", "rag_answer", "nl2sql", "chitchat"} <= names

    # 条件边分支逻辑
    assert gm._branch({"skill": "nl2sql"}) == "nl2sql"
    assert gm._branch({"skill": "chitchat"}) == "chitchat"
    assert gm._branch({"skill": "multimodal"}) == "rag"  # 读图走 RAG 文档问答边
    assert gm._branch({"skill": "rag"}) == "rag"
    assert gm._branch({}) == "rag"                        # 缺省兜底