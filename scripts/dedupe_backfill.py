#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dedupe_backfill.py — 修复 Milvus 历史块的 doc_id/category 漂移并清理重复/孤儿块。

背景（严重 bug）：
  Milvus schema 仅有 8 个静态字段，enable_dynamic_field=True。
  旧 _build_row 把 doc_id/source_file 塞进 metadata 嵌套 dict，Milvus 无法用
  metadata["doc_id"] 访问，导致 delete_chunks_by_document 的过滤永远匹配不到行、
  删除彻底失效；历史上传因此累积大量重复/孤儿向量。
  修复后 _build_row 已把 doc_id/source_file 提升为扁平动态字段，删除 filter 改用
  扁平字段。本脚本把存量历史块回填 doc_id/kb_id/source_file/category，并按 question
  去重（每 question 保留 1 份代表行，删其余），无法关联到 MySQL 文档的孤儿块直接删除。
  因 Milvus upsert 要求全字段，采用 delete 旧代表行 + insert 新行（含原向量）回填。

用法：服务运行即可（直接连 Milvus，无需重启）。
"""
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, ".")

from db.milvus import COLLECTION_NAME, ensure_collection, get_milvus_client
from db.mysql import get_db

client = get_milvus_client()
ensure_collection(client)

with get_db() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT d.id, d.kb_id, d.filename FROM documents d")
        docs = {r["id"]: {"kb_id": r["kb_id"], "filename": r["filename"]} for r in cur.fetchall()}
        cur.execute("SELECT id, domain FROM knowledge_bases")
        kb_domain = {r["id"]: (r["domain"] or "general") for r in cur.fetchall()}
doc_by_filename = defaultdict(list)
for did, d in docs.items():
    doc_by_filename[d["filename"]].append(did)
print(f"MySQL 文档数: {len(docs)}, 知识库 domain 数: {len(kb_domain)}")

rows = client.query(
    COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["id", "question", "kb_id", "doc_id", "source_file", "category", "metadata"],
    limit=16384,
)
doc_rows = [r for r in rows if (r.get("question") or "").startswith("doc_")]
print(f"Milvus 文档块总数: {len(doc_rows)} (含 FAQ 总行数 {len(rows)})")

pat = re.compile(r"^doc_(\d+)_")
groups = defaultdict(list)
for r in doc_rows:
    groups[r["question"]].append(r)

to_insert = []   # 代表行(带原向量 + 回填字段)
to_delete = []   # 旧代表行 id + 重复行 id + 孤儿行 id
plan = []        # (rep_id, rid, kb_id, fname, category, dup_ids)
orphan = 0
for q, rs in groups.items():
    rep = rs[0]
    meta = rep.get("metadata") or {}
    fname = rep.get("source_file") or meta.get("source_file") or ""
    m = pat.match(q)
    N = int(m.group(1)) if m else None
    rid = None
    if rep.get("doc_id") and rep["doc_id"] in docs:
        rid = rep["doc_id"]
    elif N is not None and N in docs:
        rid = N
    elif fname:
        cands = doc_by_filename.get(fname)
        if cands:
            rid = cands[0]
    if rid is None:
        orphan += 1
        for r in rs:
            to_delete.append(r["id"])
        continue
    dm = docs[rid]
    kb_id = dm["kb_id"]
    category = kb_domain.get(kb_id, "general")
    plan.append((rep["id"], rid, kb_id, fname, category, [r["id"] for r in rs[1:]]))
    to_delete.append(rep["id"])

print(f"可关联(回填): {len(plan)}, 孤儿(删除): {orphan}")
print(f"待 insert 代表行: {len(plan)}, 待删行(旧代表+重复+孤儿): {len(to_delete)}")

# 批量取代表行完整向量，构造回填新行
all_rep_ids = [p[0] for p in plan]
fulls = client.query(
    COLLECTION_NAME,
    filter=f'id in [{",".join(map(str, all_rep_ids))}]',
    output_fields=["*"],
    limit=8000,
)
full_map = {f["id"]: f for f in fulls}
missing = [p[0] for p in plan if p[0] not in full_map]
if missing:
    print(f"[warn] {len(missing)} 个代表行未取到完整数据，跳过: {missing[:5]}")
for rep_id, rid, kb_id, fname, category, _ in plan:
    f = full_map.get(rep_id)
    if not f:
        continue
    f["doc_id"] = rid
    f["kb_id"] = kb_id
    f["source_file"] = fname
    f["category"] = category
    f.pop("id", None)
    to_insert.append(f)

B = 500
for i in range(0, len(to_insert), B):
    client.insert(COLLECTION_NAME, data=to_insert[i:i + B])
    print(f"  [insert] {i + len(to_insert[i:i + B])}/{len(to_insert)}")
for i in range(0, len(to_delete), B):
    client.delete(COLLECTION_NAME, to_delete[i:i + B])
    print(f"  [delete] {i + len(to_delete[i:i + B])}/{len(to_delete)}")

time.sleep(4)
rows2 = client.query(COLLECTION_NAME, filter="id >= 0", output_fields=["question", "doc_id", "kb_id", "category"], limit=16384)
doc2 = [r for r in rows2 if (r.get("question") or "").startswith("doc_")]
cnt = Counter(r["question"] for r in doc2)
dups = sum(1 for v in cnt.values() if v > 1)
none_doc = sum(1 for r in doc2 if not r.get("doc_id"))
print(f"复核: 文档块行数={len(doc2)}, 重复 question 数={dups}, doc_id 为空的块={none_doc}")
print("各 kb 块数(按 kb_id):", dict(Counter(r.get("kb_id") for r in doc2)))
print("[done] 回填+清理完成。")
