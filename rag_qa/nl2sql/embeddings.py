#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/embeddings.py — 阿里云百炼 API Embedding 客户端（qwen3.7-text-embedding-flash）。

用于 Schema 向量召回；与 FinRag 本地 BGE-M3（services/embedding.py，非结构化文档）区分开。
支持 dimensions 参数，模型不认时去参重试；连接复用 services/llm_ext 的凭证与 trust_env=False 策略。
"""
from __future__ import annotations

import asyncio
import logging

import httpx
from openai import AsyncOpenAI

from config import get_config

logger = logging.getLogger(__name__)

_embed_client: AsyncOpenAI | None = None
_embed_lock = asyncio.Lock()
_embed_supports_dim: bool | None = None


async def get_embed_client() -> AsyncOpenAI:
    global _embed_client
    if _embed_client is None:
        async with _embed_lock:
            if _embed_client is None:
                cfg = get_config()
                _embed_client = AsyncOpenAI(
                    api_key=cfg.llm.api_key,
                    base_url=cfg.llm.base_url,
                    timeout=cfg.llm.timeout_seconds,
                    max_retries=cfg.llm.max_retries,
                    http_client=httpx.AsyncClient(trust_env=False, timeout=cfg.llm.timeout_seconds),
                )
    return _embed_client


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return (dot / (na * nb)) if na and nb else 0.0


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量文本 → 向量（维度由配置 embedding_dim 指定；模型不认 dimensions 时降级不带参）。

    阿里云百炼对单次入参批大小有限制（过大报 "batch size is invalid"），
    因此按小块分批调用再拼接。
    """
    global _embed_supports_dim
    if not texts:
        return []
    cfg = get_config().nl2sql
    client = await get_embed_client()
    batch = 5
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        attempts = [
            {"model": cfg.embedding_model, "input": chunk, "dimensions": cfg.embedding_dim},
            {"model": cfg.embedding_model, "input": chunk},
        ]
        if _embed_supports_dim is False:
            attempts.pop(0)
        last_err: Exception | None = None
        for kwargs in attempts:
            try:
                resp = await client.embeddings.create(**kwargs)
                _embed_supports_dim = ("dimensions" in kwargs)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                low = str(e).lower()
                if "dimensions" not in low and "parameter" not in low:
                    raise
                logger.warning("embedding dimensions 参数不被支持，去参重试: %s", str(e)[:120])
                _embed_supports_dim = False
        else:
            raise last_err or RuntimeError("embedding failed")
    return out


def rank_by_query(question_vec: list[float], entries: list[dict]) -> list[tuple[int, float]]:
    """按 question 向量与语料条目向量的余弦相似度排序，返回 (index, score)。"""
    scored = [(i, cosine(question_vec, e["vec"])) for i, e in enumerate(entries)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored