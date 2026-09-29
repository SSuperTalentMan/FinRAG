# -*- coding: utf-8 -*-
"""诊断工具：打印全部候选（BM25 + Milvus）的重排分数，用于定位排序异常。

用法：
    .venv/Scripts/python.exe scripts/probe_rerank_scores.py
改脚本内的 QUESTIONS 列表即可换诊断问题。
注意：文档块的 question 是 chunk id，在不同文档间会重复，
     回查内容时必须带 kb_id / category 过滤，否则会串到别的文档上。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config  # noqa: E402
from db.milvus import ensure_collection, get_milvus_client, search_milvus  # noqa: E402
from rag_qa.core.query_classifier import classify_intent  # noqa: E402
from services.bm25 import get_bm25_retriever  # noqa: E402
from services.embedding import encode_query_dense_sparse  # noqa: E402
from services.reranker import get_reranker, is_chunk_id, rerank_text  # noqa: E402

QUESTIONS = [
    "个人信用报告可以在哪里查询？",
    "数字人民币迎来了什么重大调整？",
    "金融科技发展的主要目标是什么？",
    "企业会计准则中收入确认的原则是什么？",
]

cfg = get_config()
client = get_milvus_client()
ensure_collection(client)
model = get_reranker()

for q in QUESTIONS:
    intent = classify_intent(q)
    dom = intent.domain if intent.domain != "general" else None
    qd, qs = encode_query_dense_sparse(q)
    milvus_hits = search_milvus(client, qd, qs, k=cfg.retrieval.retrieval_k, domain=dom)
    bm25_hits = get_bm25_retriever().search(q, domain=dom, top_k=5) or []

    cands, seen = [], set()
    for h in bm25_hits:
        if h["question"] not in seen:
            cands.append({**h, "source": h.get("source") or "bm25"})
            seen.add(h["question"])
    for h in milvus_hits:
        if h["question"] not in seen:
            cands.append({**h, "source": h.get("source", "milvus")})
            seen.add(h["question"])

    pairs = [(q, rerank_text(c)) for c in cands]
    scores = model.predict(pairs)

    print("=" * 110)
    print(f"Q: {q}  intent={intent.domain}  候选={len(cands)}（bm25={len(bm25_hits)}, milvus={len(milvus_hits)}）")
    for c, s in sorted(zip(cands, scores), key=lambda x: -float(x[1])):
        qq = c.get("question", "")
        txt = rerank_text(c)
        if is_chunk_id(qq):
            label = f"[文档块 len={len(txt)}] {txt[:65]}"
        else:
            label = f"[FAQ len={len(txt)}] {qq[:65]}"
        print(f"   {float(s):>8.4f} | {c.get('source', '')[:12]:<12} | {label}")
