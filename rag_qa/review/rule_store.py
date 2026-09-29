#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/rule_store.py — 监管规则检索器（融合 DocAudit RuleRetriever，适配 FinRag BGE-M3）。

P1 精简：风险规则库规模小（种子 20 条以内），直接加载进内存做"BGE-M3 稠密余弦 +
jieba 关键词重叠 + RRF 融合"，并对候选做交叉编码器重排（可选，本地模型存在即生效）。
避免为小规则库新开 Milvus collection。检索可信度低时允许返回空规则 → 审查判 insufficient_evidence。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import jieba

from config import get_config
from rag_qa.review import db as rdb
from services.embedding import encode_batch
from services.reranker import get_reranker

logger = logging.getLogger(__name__)


def _rrf(dense_scores: list[tuple[int, float]], kw_scores: list[tuple[int, float]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion：合并稠密与关键词排序。"""
    ranks: dict[int, int] = {}
    for source in (dense_scores, kw_scores):
        for idx, _ in sorted(source, key=lambda x: x[1], reverse=True)[:k]:
            ranks[idx] = ranks.get(idx, 0) + 1 / (60 + len(ranks))
    return [idx for idx, _ in sorted(ranks.items(), key=lambda x: x[1], reverse=True)]


class RuleRetriever:
    def __init__(self) -> None:
        self._rules: list[dict] = []
        self._rule_matrix: list[list[float]] | None = None
        self._tokens: list[set[str]] = []
        self._loaded = False

    def load_sync(self) -> None:
        """从 MySQL 加载全部规则并切词（同步，供启动预热）。"""
        rows = rdb.fetch_all(
            "SELECT rule_id, doc_name, article_no, requirement, severity, keywords "
            "FROM docaudit_risk_rule ORDER BY id"
        )
        self._rules = [
            {
                "rule_id": r["rule_id"],
                "doc_name": r["doc_name"] or "",
                "article_no": r["article_no"] or "",
                "requirement": r["requirement"] or "",
                "severity": r["severity"] or "medium",
                "keywords": r["keywords"] or "",
            }
            for r in rows
        ]
        self._tokens = [set(jieba.cut(f"{r['requirement']} {r['keywords']}")) for r in self._rules]
        self._rule_matrix = None  # 稠密向量懒加载，避免冷启动时阻塞
        self._loaded = True
        logger.info("规则加载完成: %s 条", len(self._rules))

    async def _ensure_vectors(self) -> None:
        if self._rule_matrix is not None or not self._rules:
            return
        try:
            texts = [f"{r['requirement']} {r.get('keywords') or ''}" for r in self._rules]
            self._rule_matrix = await asyncio.to_thread(encode_batch, texts)
        except Exception as e:  # noqa: BLE001
            logger.warning("规则稠密向量化失败（降级为纯关键词检索）: %s", e)
            self._rule_matrix = []

    def _keyword_scores(self, query_tokens: set[str]) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for i, toks in enumerate(self._tokens):
            if not toks:
                continue
            inter = len(query_tokens & toks)
            if inter > 0:
                out.append((i, inter / (len(toks) ** 0.5)))
        return out

    async def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[dict], dict[str, Any]]:
        """检索相关规则。返回 (规则列表, debug)。无规则库或检索为空时返回空列表。"""
        cfg = get_config()
        top_k = top_k or cfg.compliance.retrieve_top_k
        if not self._loaded:
            await asyncio.to_thread(self.load_sync)
        if not self._rules:
            return [], {"source": "empty"}

        query_tokens = set(jieba.cut(query))
        kw = self._keyword_scores(query_tokens)
        dense: list[tuple[int, float]] = []
        if self._rule_matrix:
            try:
                qvec = (await asyncio.to_thread(encode_batch, [query]))[0]
                dense = sorted(
                    ((i, sum(a * b for a, b in zip(qvec, self._rule_matrix[i])))
                     for i in range(len(self._rules))),
                    key=lambda x: x[1], reverse=True,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("规则稠密检索失败（仅用关键词）: %s", e)

        order = _rrf(dense, kw)[: max(top_k, 6)]
        cands = [self._rules[i] for i in order]

        # 交叉编码器重排（本地 bge-reranker 存在即生效），否则用 RRF 序
        try:
            reranker = get_reranker()
            pairs = [[query, r['requirement'][:2000]] for r in cands]
            scores = reranker.predict(pairs, batch_size=8).tolist()
            rank = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = [cands[i] for i in rank]
            source = "rerank"
        except Exception as e:  # noqa: BLE001
            logger.info("重排不可用（用 RRF 序返回）: %s", e)
            results = cands[:top_k]
            source = "rrf"
        return results, {"source": source, "candidates": len(cands), "rules": len(self._rules)}


_retriever: RuleRetriever | None = None


def get_rule_retriever() -> RuleRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RuleRetriever()
        try:
            _retriever.load_sync()
        except Exception as e:  # noqa: BLE001  # 表未建时不阻塞，首次 retrieve 再加载
            logger.warning("规则检索器初始化延迟（可冷启动）: %s", e)
    return _retriever