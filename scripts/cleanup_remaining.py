#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cleanup_remaining.py — 删除 Milvus 中 doc_id 为空的旧文档块（保留已回填 doc_id 的新行）。"""
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, ".")

from db.milvus import COLLECTION_NAME, ensure_collection, get_milvus_client

c = get_milvus_client()
ensure_collection(c)

rows = c.query(COLLECTION_NAME, filter="id >= 0", output_fields=["id", "question", "doc_id"], limit=16384)
doc_rows = [r for r in rows if (r.get("question") or "").startswith("doc_")]
g = defaultdict(list)
for r in doc_rows:
    g[r["question"]].append(r)

del_ids = []
for r in doc_rows:
    if not r.get("doc_id"):
        # 仅当该 question 已有带 doc_id 的行时才删（避免误删唯一块）
        if any(x.get("doc_id") for x in g[r["question"]]):
            del_ids.append(r["id"])

print(f"文档块总数 {len(doc_rows)}, 待删空 doc_id 旧行 {len(del_ids)}")
B = 1000
for i in range(0, len(del_ids), B):
    c.delete(COLLECTION_NAME, del_ids[i:i + B])
    print(f"  [delete] {i + len(del_ids[i:i + B])}/{len(del_ids)}")
time.sleep(4)

rows2 = c.query(COLLECTION_NAME, filter="id >= 0", output_fields=["question", "doc_id", "kb_id"], limit=16384)
doc2 = [r for r in rows2 if (r.get("question") or "").startswith("doc_")]
cnt = Counter(r["question"] for r in doc2)
dups = sum(1 for v in cnt.values() if v > 1)
none_doc = sum(1 for r in doc2 if not r.get("doc_id"))
print(f"复核: 文档块={len(doc2)}, 重复 question={dups}, doc_id 空块={none_doc}")
print("各 kb 块数(按 kb_id):", dict(Counter(r.get("kb_id") for r in doc2)))
print("[done] 清理完成。")
