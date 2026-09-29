#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轻量检查：Milvus 行 question 字段是否全局唯一（重建后应为 doc_{doc_id}_... 全局唯一）。"""
import sys
sys.path.insert(0, ".")

from collections import Counter

from db.milvus import COLLECTION_NAME, ensure_collection, get_milvus_client

client = get_milvus_client()
ensure_collection(client)
rows = client.query(
    COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["question", "category"],
    limit=16384,
)
# 仅统计文档块（question 以 doc_ 开头），排除 FAQ 问答对（其 question 为自然语言，可能本身重复）
doc_rows = [r for r in rows if (r["question"] or "").startswith("doc_")]
print(f"文档块总行数: {len(doc_rows)}  (Milvus 总行数含 FAQ: {len(rows)})")
questions = [r["question"] for r in doc_rows]
uniq = set(questions)
dups = len(questions) - len(uniq)
print(f"文档块唯一 question 数: {len(uniq)}")
print(f"文档块重复 question 数: {dups}")
print("文档块按 category 分布:")
for c, n in Counter(r["category"] for r in doc_rows).most_common():
    print(f"  {c}: {n}")
if dups:
    seen = set()
    print("文档块重复样例:")
    for q in questions:
        if q in seen:
            print("   ", q)
        else:
            seen.add(q)
else:
    print("OK: 文档块 question 全部唯一，chunk id 跨文件重复问题已消除。")
