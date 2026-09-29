#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精确复刻 routers/chat.py::_build_context 的检索链路，定位 doc_4 未被召回的根因。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
logger.remove(); logger.add(sys.stderr, level="WARNING")

from config import get_config
from db.milvus import get_milvus_client, search_milvus, ensure_collection
from services.embedding import encode_query_dense_sparse
from services.bm25 import get_bm25_retriever
from services.reranker import is_chunk_id, rerank
from rag_qa.core.query_classifier import classify_intent

cfg = get_config()
client = get_milvus_client()
ensure_collection(client)

QUERIES = [
    "注册制下发行人信息披露有哪些要求？",
    "股票发行注册制改革的主要内容是什么？",
    "投资者教育包括哪些内容？",
]

def build_context_like_chat(question, intent):
    """复刻 _build_context，返回中间结果便于排查。"""
    query_dense, query_sparse = encode_query_dense_sparse(question)
    milvus_domain = intent.domain if intent.domain != "general" else None
    milvus_hits = search_milvus(client, query_dense, query_sparse, k=cfg.retrieval.retrieval_k, domain=milvus_domain)
    fallback_used = False
    if not milvus_hits and milvus_domain:
        fallback_used = True
        milvus_hits = search_milvus(client, query_dense, query_sparse, k=cfg.retrieval.retrieval_k, domain=None)
    bm25_domain = intent.domain if intent.domain != "general" else None
    bm25_retriever = get_bm25_retriever()
    bm25_hits = bm25_retriever.search(question, domain=bm25_domain, top_k=5) or []
    return milvus_hits, bm25_hits, fallback_used, milvus_domain

def count_doc4(hits):
    return sum(1 for h in hits if str(h.get("question", "")).startswith("doc_4_"))

def count_doc31(hits):
    return sum(1 for h in hits if str(h.get("question", "")).startswith("doc_31_"))

for q in QUERIES:
    intent = classify_intent(q)
    mh, bh, fallback, md = build_context_like_chat(q, intent)
    d4 = count_doc4(mh); d31 = count_doc31(mh)
    print("=" * 70)
    print(f"Q: {q}")
    print(f"  classify -> domain={intent.domain} conf={intent.confidence:.4f} hits={intent.keywords_hit}")
    print(f"  milvus_domain(过滤) = {md}  fallback全库={fallback}")
    print(f"  milvus_hits={len(mh)} (doc_4={d4}, doc_31={d31})  bm25_hits={len(bh)}")
    # rerank
    seen = set()
    candidates = []
    for h in bh:
        if h["question"] not in seen:
            candidates.append({**h, "source": h.get("source") or "bm25"}); seen.add(h["question"])
    for h in mh:
        if h["question"] not in seen:
            candidates.append({"question": h["question"], "answer": h["answer"], "text": h.get("text",""),
                               "category": h["category"], "score": h["score"], "source": h.get("source","milvus")})
            seen.add(h["question"])
    top_k = min(5, len(candidates))
    reranked = rerank(q, candidates, top_k=top_k) if candidates else []
    print(f"  rerank -> top{top_k} 是否含 doc_4={any(str(r['question']).startswith('doc_4_') for r in reranked)}"
          f" 含 doc_31={any(str(r['question']).startswith('doc_31_') for r in reranked)}")
    for r in reranked:
        print(f"     - {r['question'][:45]}  score={r.get('rerank_score', r.get('score')):.4f}")

print("=" * 70)
print("【对照】强制 domain=investment_banking 检索同一批问题：")
for q in QUERIES:
    qd, qs = encode_query_dense_sparse(q)
    hits = search_milvus(client, qd, qs, k=cfg.retrieval.retrieval_k, domain="investment_banking")
    print(f"  Q={q[:18]}...  investment_banking命中={len(hits)} (doc_4={count_doc4(hits)}, doc_31={count_doc31(hits)})")
