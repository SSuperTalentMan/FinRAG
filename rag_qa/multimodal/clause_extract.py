#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/multimodal/clause_extract.py — 条款结构化抽取：LLM 抽取 + 三重校验（页码回验/编号连续/逐字覆盖）+ 失败回喂重试。"""
from __future__ import annotations

import logging
import random
import re

from config import get_config
from rag_qa.review.schemas import ParseUnit
from rag_qa.review.prompts import EXTRACT_SYSTEM, EXTRACT_RETRY
from services.llm_ext import chat_json

logger = logging.getLogger(__name__)

RE_CLAUSE_NO = re.compile(r"第[一二三四五六七八九十百零\d]+条")


def shingle_coverage(content: str, page_text: str, probes: int = 5, shingle: int = 12) -> float:
    """页码回验：从条款内容取若干 shingle，检查是否在对应页原文中出现，返回命中率。"""
    clean = re.sub(r"\s+", "", content)
    clean_page = re.sub(r"\s+", "", page_text)
    if len(clean) < shingle:
        return 1.0 if clean and clean in clean_page else 0.0
    starts = random.Random(42).sample(range(len(clean) - shingle + 1), k=min(probes, len(clean) - shingle + 1))
    hits = sum(1 for st in starts if clean[st:st + shingle] in clean_page)
    return hits / len(starts)


def check_continuity(clause_nos: list[str]) -> list[str]:
    """编号连续性：解析"第X条"为整数序列，报告缺失/重复/无法解析的缺陷。"""
    cn_map = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

    def cn2int(s: str) -> int | None:
        s = s.replace("第", "").replace("条", "")
        if s.isdigit():
            return int(s)
        if s == "十":
            return 10
        if "十" in s:
            left, _, right = s.partition("十")
            total = (cn_map.get(left, 1) if left else 1) * 10 + (cn_map.get(right, 0) if right else 0)
            return total if total > 0 else None
        if all(ch in cn_map for ch in s) and s:
            return sum(cn_map[ch] * (10 ** (len(s) - 1 - i)) for i, ch in enumerate(s)) if len(s) > 1 else cn_map.get(s)
        return None

    defects: list[str] = []
    nums: list[int] = []
    for no in clause_nos:
        m = RE_CLAUSE_NO.search(no or "")
        n = cn2int(m.group()) if m else None
        if n is None:
            defects.append(f"无法解析条款号: {no}")
        else:
            nums.append(n)
    if nums:
        expected = list(range(nums[0], nums[0] + len(nums)))
        missing = sorted(set(expected) - set(nums))
        duplicated = sorted({n for n in nums if nums.count(n) > 1})
        if missing:
            defects.append(f"编号缺失: {missing}")
        if duplicated:
            defects.append(f"编号重复: {duplicated}")
    return defects


def build_extract_input(units: list[ParseUnit]) -> str:
    """解析单元 → 带页码标记的输入文本。"""
    parts, cur_page = [], None
    for u in units:
        if u.page_no != cur_page:
            parts.append(f"\n【第 {u.page_no} 页】")
            cur_page = u.page_no
        prefix = {"title": "【标题】", "table": "【表格】", "figure": "【图片】"}.get(u.block_type, "")
        parts.append(prefix + u.text)
    return "\n".join(parts)


def _parse_clauses(data: dict) -> list[dict]:
    out: list[dict] = []
    for c in data.get("clauses") or []:
        no = str(c.get("clause_no", "")).strip()
        content = str(c.get("content", "")).strip()
        if not no or not content:
            continue
        out.append({
            "clause_no": no,
            "title": str(c.get("title", "")).strip()[:200],
            "content": content,
            "page_start": int(c.get("page_start", 1) or 1),
            "page_end": int(c.get("page_end", c.get("page_start", 1)) or 1),
        })
    return out


def validate(clauses: list[dict], page_texts: dict[int, str], coverage_threshold: float) -> tuple[list[dict], list[str]]:
    """三重校验，返回 (条款列表, 缺陷列表)。"""
    defects: list[str] = []
    for c in clauses:
        problems: list[str] = []
        for pg in {c["page_start"], c["page_end"]}:
            pt = page_texts.get(pg)
            if pt is None:
                problems.append(f"页码 {pg} 不存在")
                continue
            cov = shingle_coverage(c["content"], pt)
            if cov < coverage_threshold:
                problems.append(f"第{pg}页内容覆盖率 {cov:.0%}")
        if not RE_CLAUSE_NO.search(c["clause_no"]):
            problems.append(f"条款号非标准格式: {c['clause_no']}")
        status = "ok" if not problems else ("failed" if len(problems) >= 2 else "partial")
        if problems:
            defects.append(f"{c['clause_no']}: {';'.join(problems)}")
        c["extract_status"] = status
        c["defects"] = ";".join(problems)[:500]
    defects.extend(check_continuity([c["clause_no"] for c in clauses]))
    return clauses, defects


async def extract_clauses(units: list[ParseUnit], page_texts: dict[int, str]) -> tuple[list[dict], list[str]]:
    """抽取 + 校验 + 回喂重试（≤ max_retries）。"""
    cfg = get_config()
    doc_input = build_extract_input(units)
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": doc_input},
    ]
    clauses: list[dict] = []
    defects: list[str] = ["未生成"]
    for attempt in range(cfg.compliance.extract_max_retries + 1):
        try:
            data, _ = await chat_json(messages, model=cfg.llm.sql_model or None, stage="extract")
            clauses = _parse_clauses(data)
        except Exception as e:  # noqa: BLE001
            logger.warning("extract attempt %s failed: %s", attempt + 1, e)
            clauses = []
        clauses, defects = validate(clauses, page_texts, cfg.compliance.coverage_threshold)
        hard = [d for d in defects if "缺失" in d or "重复" in d or "覆盖率" in d or "不存在" in d]
        if clauses and not hard:
            return clauses, defects
        if attempt < cfg.compliance.extract_max_retries:
            failure = "\n".join(defects) if defects else "未抽取到任何条款"
            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": EXTRACT_RETRY.format(failure=failure, input=doc_input)},
            ]
    return clauses, defects