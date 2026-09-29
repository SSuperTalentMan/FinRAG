#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_retrieval_kb4.py — 检索实证：kb=4（investment_banking 域）内
混合检索能否召回 doc_4（注册制 PDF）与 doc_31（投资者教育文章.pdf）的内容。
"""
import sys
sys.path.insert(0, ".")

from db.milvus import get_milvus_client, search_milvus
from services.embedding import encode_query_dense_sparse

client = get_milvus_client()
DOMAIN = "investment_banking"

QUERIES = [
    ("注册制下发行人信息披露要求", "doc_4_"),
    ("投资者保护要点与维权渠道", "doc_4_"),
    ("股票发行注册制改革审核注册流程", "doc_4_"),
    ("投资者教育：识别非法证券活动", "doc_31_"),
    ("开户风险揭示与适当性管理", "doc_31_"),
]

print(f"=== kb=4 域({DOMAIN}) 混合检索实证 ===")
all_ok = True
for q, prefix in QUERIES:
    dense, sparse = encode_query_dense_sparse(q)
    res = search_milvus(client, dense, sparse, k=13, domain=DOMAIN)
    hit = [r for r in res if str(r.get("question", "")).startswith(prefix)]
    ok = len(hit) >= 1
    all_ok = all_ok and ok
    print(f"  [{'OK' if ok else 'MISS'}] 「{q}」Top{len(res)}: 命中 {prefix} 块 {len(hit)} 条")
    for r in res[:3]:
        print(f"        - {r.get('question')}  score={r.get('score'):.3f} src={r.get('source')}")

print("\n=== 结论 ===")
print("全部查询均可召回目标文档内容" if all_ok else "存在未召回目标文档的查询，需排查")
