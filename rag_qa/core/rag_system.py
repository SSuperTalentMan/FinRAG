#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/core/rag_system.py — 多策略 RAG 检索系统
支持：直接检索 / HyDE / 子查询 / 回溯问题 四种检索策略。
"""

from loguru import logger
from config import get_config
from db.milvus import get_milvus_client, search_milvus, ensure_collection
from services.bm25 import get_bm25_retriever
from services.reranker import rerank
from services.embedding import encode_query_dense_sparse
from services.llm import chat_completion
from .prompts import RAGPrompts
from .strategy_selector import (
    StrategySelector,
    STRATEGY_DIRECT,
    STRATEGY_HYDE,
    STRATEGY_SUBQUERY,
    STRATEGY_BACKTRACK,
)


class RAGSystem:
    """
    多策略 RAG 检索系统。
    支持四种检索策略，最终通过 BGE-Reranker 精排返回 Top-K 上下文。
    """

    def __init__(self):
        self.config = get_config()
        self.strategy_selector = StrategySelector()

    # ─── 检索核心 ────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, strategy: str = STRATEGY_DIRECT,
                 domain: str | None = None, k: int = 5) -> list[dict]:
        """
        根据策略执行检索，返回合并后的候选列表（每项含 question/answer/category/score/source）。
        """
        strategy = strategy or STRATEGY_DIRECT

        if strategy == STRATEGY_HYDE:
            results = self._retrieve_hyde(query, domain, k)
        elif strategy == STRATEGY_SUBQUERY:
            results = self._retrieve_subquery(query, domain, k)
        elif strategy == STRATEGY_BACKTRACK:
            results = self._retrieve_backtrack(query, domain, k)
        else:
            results = self._retrieve_direct(query, domain, k)

        logger.debug(f"策略 [{strategy}] 检索到 {len(results)} 条候选")
        return results

    def _retrieve_direct(self, query: str, domain: str | None, k: int) -> list[dict]:
        """直接检索：Milvus 向量 + BM25 融合。"""
        return self._hybrid_search(query, domain, k)

    def _retrieve_hyde(self, query: str, domain: str | None, k: int) -> list[dict]:
        """HyDE：先用 LLM 生成假设答案，再基于假设答案做向量检索。"""
        logger.info(f"使用 HyDE 策略 (查询: '{query}')")
        try:
            hypo_prompt = RAGPrompts.hyde_prompt(query)
            hypo_answer = chat_completion(
                messages=[{"role": "user", "content": hypo_prompt}], stream=False
            )
            logger.info(f"HyDE 假设答案: {hypo_answer[:100]}...")
            return self._hybrid_search(hypo_answer, domain, k)
        except Exception as e:
            logger.error(f"HyDE 策略执行失败，降级为直接检索: {e}")
            return self._hybrid_search(query, domain, k)

    def _retrieve_subquery(self, query: str, domain: str | None, k: int) -> list[dict]:
        """子查询检索：将复杂问题拆分为多个子查询，分别检索后合并去重。"""
        logger.info(f"使用子查询策略 (查询: '{query}')")
        try:
            sub_prompt = RAGPrompts.subquery_prompt(query)
            sub_text = chat_completion(
                messages=[{"role": "user", "content": sub_prompt}], stream=False
            )
            sub_queries = [q.strip() for q in sub_text.strip().split("\n") if q.strip()]
            logger.info(f"生成的子查询: {sub_queries}")
            if not sub_queries:
                return self._hybrid_search(query, domain, k)

            all_results: dict[str, dict] = {}
            for sq in sub_queries:
                docs = self._hybrid_search(sq, domain, k)
                for d in docs:
                    key = d["question"]
                    if key not in all_results:
                        all_results[key] = d
                    else:
                        # 保留最高分数
                        if d.get("score", 0) > all_results[key].get("score", 0):
                            all_results[key] = d

            results = list(all_results.values())
            logger.info(f"子查询合并后共 {len(results)} 条候选")
            return results
        except Exception as e:
            logger.error(f"子查询策略执行失败，降级为直接检索: {e}")
            return self._hybrid_search(query, domain, k)

    def _retrieve_backtrack(self, query: str, domain: str | None, k: int) -> list[dict]:
        """回溯检索：将复杂问题简化后检索。"""
        logger.info(f"使用回溯策略 (查询: '{query}')")
        try:
            bt_prompt = RAGPrompts.backtracking_prompt(query)
            simplified = chat_completion(
                messages=[{"role": "user", "content": bt_prompt}], stream=False
            ).strip()
            logger.info(f"简化后的问题: '{simplified}'")
            return self._hybrid_search(simplified, domain, k)
        except Exception as e:
            logger.error(f"回溯策略执行失败，降级为直接检索: {e}")
            return self._hybrid_search(query, domain, k)

    # ─── 混合检索 ────────────────────────────────────────────────────────────────
    def _hybrid_search(self, query: str, domain: str | None, k: int) -> list[dict]:
        """稠密+稀疏混合检索（Milvus）+ BM25 召回，融合去重后返回候选。"""
        client = get_milvus_client()
        ensure_collection(client)
        query_dense, query_sparse = encode_query_dense_sparse(query)

        # Milvus 混合检索（稠密 + 稀疏）
        milvus_domain = domain if domain and domain != "general" else None
        milvus_hits = search_milvus(client, query_dense, query_sparse, k=k, domain=milvus_domain)

        # BM25 召回
        bm25 = get_bm25_retriever()
        bm25_hits = bm25.search(query, domain=domain, top_k=k) or []

        # Milvus 实体未存 source，用 question 回查 MySQL 补全来源（便于答案标注出处）
        from db.mysql import get_question_source_map
        src_map = get_question_source_map()

        # 融合去重
        candidates: dict[str, dict] = {}
        for hit in bm25_hits:
            q = hit["question"]
            if q not in candidates:
                candidates[q] = {
                    "question": q,
                    "answer": hit["answer"],
                    "category": hit["category"],
                    "score": hit["softmax_score"],
                    "source": hit.get("source", ""),
                }
        for hit in milvus_hits:
            q = hit["question"]
            if q not in candidates:
                src = src_map.get(q, {})
                candidates[q] = {
                    "question": q,
                    "answer": hit["answer"],
                    "category": hit["category"],
                    "score": hit["score"],
                    "source": src.get("source") or hit.get("source", "milvus"),
                    "source_url": src.get("source_url", ""),
                }
        return list(candidates.values())

    # ─── 精排 ────────────────────────────────────────────────────────────────────
    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """对候选列表执行 BGE-Reranker 精排，返回 Top-K。"""
        if not candidates:
            return []
        return rerank(query, candidates, top_k=top_k)

    # ─── 构建上下文 ──────────────────────────────────────────────────────────────
    def build_context(self, query: str, strategy: str = STRATEGY_DIRECT,
                      domain: str | None = None, top_k: int = 5) -> tuple[str, list[dict]]:
        """
        完整检索管线：策略选择 → 检索 → 精排 → 构建上下文字符串。
        返回 (context_str, sources_list)。
        """
        # 自动选择策略（若未指定）
        if not strategy or strategy == STRATEGY_DIRECT:
            strategy = self.strategy_selector.select_strategy(query)

        candidates = self.retrieve(query, strategy=strategy, domain=domain, k=top_k * 2)
        ranked = self.rerank(query, candidates, top_k=top_k)

        context = "\n\n".join(
            f"【问题】{r['question']}\n【回答】{r['answer']}"
            for r in ranked
        )
        sources = [
            {"question": r["question"], "score": round(r.get("rerank_score", r["score"]), 4),
             "source": r["source"]}
            for r in ranked
        ]
        return context, sources
