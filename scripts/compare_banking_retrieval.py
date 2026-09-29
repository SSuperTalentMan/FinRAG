#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
非破坏对比：banking 孤儿向量是否干扰中文 banking 检索。

做法：
  - 取 MySQL banking 的 question 集合（真值）。
  - query Milvus banking 全量 → 计算孤儿 id 集合（question 不在 MySQL banking）。
  - 对若干代表性中文 banking 问法，跑生产 search(domain="banking") 取 Top-8：
      * 记录命中（question/score/category/source）
      * 标记其中是否有孤儿 id 命中（即英文印度银行史内容是否冒泡到中文检索）
  - 结论：若 Top-N 基本为中文相关且孤儿零命中 → 删除收益有限；若孤儿频繁冒泡 → 删除有价值。
不写入/不删除任何数据。
"""
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from pymilvus import MilvusClient
from config import get_config
from rag_qa.core.query_classifier import classify_intent
from routers.chat import encode_query_dense_sparse
from db.milvus import get_milvus_client, search_milvus

BANKING_QUERIES = [
    "什么是普惠金融？",
    "银行如何识别和处理问题贷款的早期预警信号？",
    "小微企业融资难怎么解决？",
    "存款利率市场化是什么意思？",
    "数字人民币对商业银行有什么影响？",
    "商业银行资本充足率监管要求是什么？",
]


def main():
    cfg = get_config()
    # MySQL banking questions
    conn = pymysql.connect(host=cfg.mysql.host, port=cfg.mysql.port, user=cfg.mysql.user,
                           password=cfg.mysql.password, database=cfg.mysql.database,
                           charset=cfg.mysql.charset, cursorclass=pymysql.cursors.DictCursor)
    try:
        cur = conn.cursor()
        cur.execute("SELECT question FROM finance_faq WHERE category='banking'")
        mysql_q = {r["question"] for r in cur.fetchall()}
    finally:
        conn.close()

    kw = dict(uri=cfg.milvus.uri, token=cfg.milvus.token) if getattr(cfg.milvus, "token", None) else dict(uri=cfg.milvus.uri)
    client = MilvusClient(**kw)
    res = client.query(collection_name=cfg.milvus.collection_name,
                       filter='category == "banking"', output_fields=["id", "question"], limit=16384)
    orphan_ids = {r["id"] for r in res if r.get("question", "") not in mysql_q}
    print(f"Milvus banking 向量: {len(res)} | 孤儿 id 数: {len(orphan_ids)}")

    client2 = get_milvus_client()
    total_orphan_hits = 0
    print("\n" + "=" * 70)
    print("删前捕获：中文 banking 查询 Top-8（标记是否命中孤儿向量）")
    print("=" * 70)
    for q in BANKING_QUERIES:
        dense, sparse = encode_query_dense_sparse(q)
        hits = search_milvus(client2, dense, sparse, k=8, domain="banking")
        oh = sum(1 for h in hits if h.get("id") in orphan_ids)
        total_orphan_hits += oh
        print(f"\n● {q}")
        print(f"  孤儿命中数: {oh}/8")
        for i, h in enumerate(hits[:8]):
            tag = "  ←孤儿" if h.get("id") in orphan_ids else ""
            qtext = (h.get("question") or "")[:42]
            # 粗判是否英文/印度内容
            eng = sum(1 for c in qtext if ord(c) < 128) / max(1, len(qtext))
            ind = "【印/英】" if eng > 0.4 else ""
            print(f"  #{i+1} s={h.get('score'):.3f} {ind}{qtext}{tag}")

    print("\n" + "=" * 70)
    print(f"汇总：6 个中文 banking 查询共命中孤儿向量 {total_orphan_hits} 次（满分 48）")
    if total_orphan_hits == 0:
        print("  → 孤儿英文向量未冒泡到中文 banking 检索 Top-8，当前不干扰；删除收益有限（仅清理存储/整洁度）。")
    else:
        print("  → 孤儿英文向量已干扰中文 banking 检索，删除有明确收益；建议建测试集合做删后复测再定。")


if __name__ == "__main__":
    main()
