#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cleanup_orphans.py — 删除 Milvus 中所有「chunk id 前缀指向已不存在的 MySQL 文档」的孤儿块。

背景：旧的删除 bug（doc_id 嵌套不可查）导致历史上传的文档从 MySQL 删除后，其 Milvus 向量
始终删不掉，累积成大量孤儿块（doc_0/doc_1/.../doc_214 等）。这些块会污染检索（召回无关旧内容）。
现在 13 个有效文档（doc_id ∈ {4,10,21..31}）都已用修复后代码重建、前缀唯一，故所有其他
doc_N 前缀的块都是孤儿，可安全删除。

安全边界（极重要）：
  - 只删除 question 前缀 doc_N_ 中 N 不在有效文档集合的块；绝不动有效文档。
  - 锚定模式用 doc_{n}_parent%（parent 紧跟 doc_id），禁止裸 doc_{n}_%：
    Milvus LIKE 的 '_' 是单字符通配符，裸 doc_1_% 会同时命中 doc_10_%/doc_11_%，
    删除 doc_1 时会连带删掉有效文档 doc_10/doc_11…（灾难性）。
"""
import sys
sys.path.insert(0, ".")
from db.milvus import get_milvus_client
from db.chunk import _delete_expr_until_empty
from config import get_config
import pymysql

client = get_milvus_client()
COLL = "finrag_faq"
cfg = get_config()
m = cfg.mysql
conn = pymysql.connect(host=m.host, port=m.port, user=m.user, password=m.password, database=m.database)
cur = conn.cursor()
cur.execute("SELECT id FROM documents")
valid = set(r[0] for r in cur.fetchall())
conn.close()

MAXN = 400
total_deleted = 0
deleted_prefixes = []
for n in range(0, MAXN + 1):
    if n in valid:
        continue
    expr = f'question like "doc_{n}_parent%"'
    probe = client.query(COLL, expr, output_fields=["question"], limit=1)
    if not probe:
        continue
    cnt = _delete_expr_until_empty(None, expr)
    total_deleted += cnt
    deleted_prefixes.append((n, cnt))
    if len(deleted_prefixes) <= 10 or n % 50 == 0:
        print(f"  删除 doc_{n}_*: {cnt} 条")

print(f"\n[done] 共删除孤儿块 {total_deleted} 条，涉及 {len(deleted_prefixes)} 个前缀")
