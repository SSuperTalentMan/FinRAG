#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/reranker.py — BGE-Reranker-V2-M3 重排序服务
对检索候选进行精排，使用 CrossEncoder 计算相关性分数。
"""

import re
import threading

from sentence_transformers import CrossEncoder
from loguru import logger
from config import get_config

_model: CrossEncoder | None = None
_model_lock = threading.Lock()  # 双检锁：避免并发首请求重复加载模型

# 文档块写入 Milvus 时，question 字段存的是确定性 chunk id
# （形如 doc_0_parent_0_child_0，见 db/chunk.py::upsert_chunks），正文在 answer。
# 这类 id 对重排毫无语义，必须改用正文参与相关性计算。
CHUNK_ID_RE = re.compile(r"^doc_\d+_parent_\d+_child_\d+$")
# 参与重排的文本最大长度（BGE-Reranker 上限 512 token，正文取前段即可）
RERANK_TEXT_MAX_CHARS = 1500


def is_chunk_id(text: str) -> bool:
    """判断 question 是否为无语义的 chunk id。"""
    return bool(text) and bool(CHUNK_ID_RE.match(text.strip()))


def rerank_text(candidate: dict) -> str:
    """
    构造参与重排的文本。优先级：
      1. 文档块：用 text（子块原文 = 向量化时的文本）。
         不能用 answer（父块全文）—— 长文本会让 BGE-Reranker 分数饱和到 ~0.99，
         导致内容无关的块霸榜，真正相关的块反而被挤出 Top-K。
      2. FAQ：question + answer（question 已是完整问题，有语义）。
    """
    q = (candidate.get("question") or "").strip()
    if is_chunk_id(q):
        t = (candidate.get("text") or "").strip()
        if t:
            return t[:RERANK_TEXT_MAX_CHARS]
        return (candidate.get("answer") or "")[:RERANK_TEXT_MAX_CHARS]

    a = (candidate.get("answer") or "").strip()
    text = f"{q}\n{a}" if a else q
    return text[:RERANK_TEXT_MAX_CHARS]


def get_reranker() -> CrossEncoder:
    """懒加载 BGE-Reranker 模型（单例，线程安全）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                cfg = get_config()
                logger.info(f"加载 BGE-Reranker: {cfg.bge_reranker_path}")
                _model = CrossEncoder(cfg.bge_reranker_path)
                logger.info("BGE-Reranker 模型加载完成")
    return _model


def is_model_loaded() -> bool:
    """BGE-Reranker 模型是否已加载（供健康检查只读探测，不触发加载）。"""
    return _model is not None


# 权威来源偏置：相关性接近时优先采用官方来源（中国政府网 / 证监会等），
# 避免翻译 FAQ 等非权威内容排到政策解读之前。仅在 rerank 分数接近时起作用
# （绝对加分很小），不会推翻明显更相关的非权威结果。
AUTHORITY_KEYWORDS = (
    "中国政府网", "中国证监会", "中国人民银行", "国家金融监督管理总局",
    "财政部", "国家统计局", "国务院", "银保监会", "发展改革委",
)
AUTHORITY_BONUS = 0.03


def _is_authoritative(source: str) -> bool:
    src = source or ""
    return any(k in src for k in AUTHORITY_KEYWORDS)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    对候选列表执行重排序。

    Parameters
    ----------
    query : str
        用户原始查询。
    candidates : list[dict]
        候选列表，每项含 question / answer / category / score / source。
    top_k : int
        重排序后保留的条数。

    Returns
    -------
    list[dict]
        按（rerank_score + 权威来源偏置）降序排列的 Top-K 结果。
    """
    # 候选不足 top_k：无需调用模型，直接按原分数（叠加权威偏置）排序返回
    if len(candidates) <= top_k:
        return sorted(
            candidates,
            key=lambda x: x.get("score", 0) + (AUTHORITY_BONUS if _is_authoritative(x.get("source", "")) else 0),
            reverse=True,
        )[:top_k]

    # 注意：不能直接用 c["question"]——文档块的 question 是 chunk id，无语义，
    # 会导致上传的文档在重排阶段被系统性淘汰。统一走 rerank_text() 取可比较文本。
    pairs = [(query, rerank_text(c)) for c in candidates]
    scores = get_reranker().predict(pairs)

    for c, s in zip(candidates, scores):
        score = float(s)
        if _is_authoritative(c.get("source", "")):
            score += AUTHORITY_BONUS
        c["rerank_score"] = score

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    logger.debug(f"Reranker 精排完成: {len(candidates)} → Top-{top_k}（含权威偏置）")
    return ranked[:top_k]
