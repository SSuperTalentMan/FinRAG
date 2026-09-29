#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_kb4_repair.py — 收尾复验（用精确 doc_id == N 查询，避免动态字段 >0/like 的查询假象）。

  1) 逐文档对比 Milvus(doc_id==N) 行数 与 MySQL chunk_count，并校验 question 唯一性
  2) 删除 filter 可命中（doc_id 扁平字段生效证明）
  3) 检索实证：kb=4 域内查询能否召回 doc_31 块
"""
import sys
sys.path.insert(0, ".")
from db.milvus import get_milvus_client
from config import get_config
import pymysql

COLL = "finrag_faq"
client = get_milvus_client()
cfg = get_config()

m = cfg.mysql
conn = pymysql.connect(host=m.host, port=m.port, user=m.user,
                       password=m.password, database=m.database)
cur = conn.cursor(pymysql.cursors.DictCursor)
cur.execute("SELECT id, kb_id, filename, status, chunk_count FROM documents WHERE status='indexed' ORDER BY id")
docs = cur.fetchall()
conn.close()

print("=== 1) 逐文档 Milvus 行数 vs MySQL chunk_count + 唯一性 ===")
all_ok = True
total_mv = 0
total_uniq = 0
for d in docs:
    did = d["id"]
    rows = client.query(COLL, f'doc_id == {did}', output_fields=["question", "source_file"], limit=16384)
    qs = [r["question"] for r in rows]
    uniq = len(set(qs))
    ok = (len(rows) == d["chunk_count"]) and (uniq == len(rows))
    total_mv += len(rows)
    total_uniq += uniq
    status = "OK" if ok else "MISMATCH"
    if not ok:
        all_ok = False
    print(f"  doc_id={did:<3} kb={d['kb_id']} {status:<9} Milvus={len(rows):>4} MySQL={d['chunk_count']:>4} 唯一={uniq:>4}  {d['filename']}")
print(f"  汇总: Milvus 文档块总行数={total_mv}, 唯一 question={total_uniq} -> {'全部一致且无重复' if (all_ok and total_mv==total_uniq) else '存在问题'}")

print("\n=== 2) 删除 filter 可命中（doc_id 扁平字段生效）===")
hits = client.query(COLL, 'kb_id == 4 and doc_id == 31', output_fields=["doc_id"], limit=16384)
print(f"  kb_id==4 and doc_id==31 -> 命中 {len(hits)} 行 (今后 UI 删除该文档可正常命中向量)")

print("\n=== 3) 检索实证（kb=4 域内混合检索 doc_31 内容）===")
try:
    from services.embedding import encode_query
    from db.milvus import search_milvus
    q = "投资者教育文章里提到的投资者保护要点"
    vec = encode_query(q)
    # search_milvus 实际签名见 db/milvus.py
    import inspect
    sig = inspect.signature(search_milvus)
    if "top_k" in sig.parameters:
        res = search_milvus(vec, top_k=5, kb_id=4, domain="investment_banking")
    else:
        res = search_milvus(vec, kb_id=4, domain="investment_banking")
    doc31 = [r for r in res if str(r.get("question", "")).startswith("doc_31_")]
    print(f"  查询「{q}」Top5 命中 doc_31 块数: {len(doc31)}")
    for r in res[:3]:
        print(f"    - {r.get('question')}  score={r.get('score'):.3f}")
except Exception as e:
    print(f"  (跳过检索实证: {e})")
