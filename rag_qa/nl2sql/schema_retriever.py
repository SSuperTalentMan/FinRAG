#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/schema_retriever.py — Schema 召回（表/字段/指标）。

策略：关键词匹配为主、向量检索（qwen3.7-text-embedding-flash）为辅的混合召回；
任何策略结果都受「按角色的表级白名单」过滤，防止越权表结构进入 Prompt。
向量召回依赖 API，失败自动降级为纯关键词。
"""
from __future__ import annotations

import asyncio
import logging

from config import get_config
from rag_qa.nl2sql import embeddings
from rag_qa.nl2sql.schema import SchemaContext

logger = logging.getLogger(__name__)

# 关键词召回：问题中命中这些词 → 优先召回对应表（无需向量也可工作）
_TABLE_KW_HINTS = {
    "门店": "dim_store", "城市": "dim_store", "大区": "dim_store", "区域": "dim_store",
    "商品": ["dim_product", "fact_order_items"], "类目": ["dim_product"],
    "客户": "dim_customer", "会员": "dim_customer", "顾客": "dim_customer",
    "日期": "dim_date", "天气": "dim_weather", "宏观": "dim_macro_cn",
    "GDP": "dim_macro_cn", "CPI": "dim_macro_cn",
    "订单": ["fact_orders", "fact_order_items"], "销售额": "fact_orders", "GMV": "fact_orders",
    "退货": "fact_refunds", "退款": "fact_refunds",
}


def _kw_hint_tables(question: str) -> set[str]:
    hits: set[str] = set()
    for kw, tables in _TABLE_KW_HINTS.items():
        if kw.lower() in question.lower():
            if isinstance(tables, str):
                hits.add(tables)
            else:
                hits.update(tables)
    return hits


class SchemaRetriever:
    def __init__(self, meta_store, top_k: int | None = None):
        self.meta = meta_store
        self.top_k = top_k or get_config().nl2sql.schema_top_k
        self._corpus_entries: list[dict] = []
        self._corpus_vecs: list[list[float]] | None = None

    def allowed_tables_for(self, role: str = "user") -> set[str]:
        """按角色的表级白名单。admin 放行全部启用表；其余要求 role 或其等价项在 allow_roles 内。"""
        all_tables = set(self.meta.enabled_tables())
        if role == "admin":
            return all_tables
        allowed: set[str] = set()
        role_equiv = {"user": "viewer", "analyst": "analyst", "admin": "admin"}
        need = role_equiv.get(role, "viewer")
        for r in self.meta.table_registry_rows:
            roles_str = (r.get("allow_roles") or "")
            roles = {x.strip().lower() for x in roles_str.split(",") if x.strip()}
            if not roles or need in roles or "viewer" in roles:
                allowed.add(r["table_name"])
        return allowed

    async def _embed_corpus_once(self) -> bool:
        """第一次调用时对全部 Schema 语料向量化并缓存。失败返回 False（走关键词）。"""
        if self._corpus_vecs is not None:
            return True
        if not self._corpus_entries:
            self._corpus_entries = self.meta.build_schema_entries()
            if not self._corpus_entries:
                return False
        try:
            texts = [e["text"] for e in self._corpus_entries]
            self._corpus_vecs = await embeddings.embed_texts(texts)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Schema 向量化失败，降级关键词召回: %s", str(e)[:120])
            return False

    async def _recall_tables(self, question: str, allowed: set[str]) -> list[str]:
        kw = _kw_hint_tables(question) & allowed
        if not get_config().nl2sql.use_embedding_recall:
            return list(kw)[: self.top_k] if kw else []

        embodied = await self._embed_corpus_once()
        if not embodied:
            return list(kw)[: self.top_k] if kw else []

        try:
            qvec = (await embeddings.embed_texts([question]))[0]
        except Exception as e:  # noqa: BLE001
            logger.warning("问题向量化失败，降级关键词: %s", str(e)[:120])
            return list(kw)[: self.top_k] if kw else []

        # 按表名聚合语料条目的最高相似度，再受白名单过滤
        table_score: dict[str, float] = {}
        for idx, entry in enumerate(self._corpus_entries):
            if entry["type"] != "table":
                continue
            tname = entry["name"]
            if tname not in allowed:
                continue
            vec = self._corpus_vecs[idx]
            table_score[tname] = max(table_score.get(tname, 0.0), embeddings.cosine(qvec, vec))
        ordered = [t for t, _ in sorted(table_score.items(), key=lambda x: x[1], reverse=True)]
        # 关键词命中的表必须进上下文（它们往往是最核心的事实表，如 GMV→fact_orders），
        # 向量结果只做补位，避免被相似度噪声挤出。
        result = list(dict.fromkeys(list(kw) + ordered))[: self.top_k]
        return result

    async def retrieve(self, question: str, role: str = "user") -> SchemaContext:
        await self.meta.ensure_loaded()
        allowed = self.allowed_tables_for(role)
        tables = await self._recall_tables(question, allowed)
        ddl_texts = [self.meta.build_ddl_text(t) for t in tables]
        metrics = self.meta.match_metrics(question)
        return SchemaContext(
            question=question, ddl_texts=ddl_texts, table_names=tables,
            metrics=metrics,
            recall_source="embedding" if self._corpus_vecs is not None else "keyword",
        )