#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/hitl.py — HITL 状态机（状态持久化 MySQL，重启可恢复）。

pending_review → approved / rejected
rejected       → 重审后回到 pending_review（re-review 由服务层触发）
全部高危条款复核完成 → 报告定稿（finalized=1）
"""
from __future__ import annotations

from datetime import datetime

VALID_TRANSITIONS = {
    None: {"pending_review"},
    "pending_review": {"approved", "rejected"},
    "rejected": {"pending_review"},
    "approved": set(),
}


def can_transition(current: str | None, target: str) -> bool:
    return target in VALID_TRANSITIONS.get(current, set())


def finalize_check(reviews: list[dict]) -> bool:
    """所有 pending 清零即可定稿。"""
    return all(r.get("hitl_status") != "pending_review" for r in reviews)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")