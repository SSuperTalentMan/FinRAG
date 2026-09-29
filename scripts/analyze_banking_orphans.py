#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只读分析：Milvus 中 banking 域的孤儿/重复向量（为清理决策提供依据，不删除任何数据）。

方法：
  - MySQL finance_faq 中 banking 的 question 集合（真值，620 条级）。
  - Milvus finrag_faq 中 category=="banking" 的全部向量（query 对 banking 可靠，属可见域）。
  - 孤儿 = Milvus banking 中 question 不在 MySQL banking 集合内的向量。
  - 重复 = Milvus banking 内同一 question 出现多次（超出首次计为可删）。
输出：统计 + 抽样 + logs/analyze_banking_orphans_<ts>.json。
"""
import sys
import os
import json
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from pymilvus import MilvusClient
from config import get_config


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

    # Milvus banking
    kw = dict(uri=cfg.milvus.uri, token=cfg.milvus.token) if getattr(cfg.milvus, "token", None) else dict(uri=cfg.milvus.uri)
    client = MilvusClient(**kw)
    res = client.query(collection_name=cfg.milvus.collection_name,
                       filter='category == "banking"', output_fields=["id", "question"], limit=16384)
    mv_q = [r.get("question", "") for r in res]
    mv_total = len(mv_q)

    mv_counter = Counter(mv_q)
    in_both = sum(1 for q in mv_q if q in mysql_q)
    orphan = [r for r in res if r.get("question", "") not in mysql_q]
    dup_extra = sum(c - 1 for c in mv_counter.values() if c > 1)  # 同一 question 重复出现超出首次的部分
    mysql_total = len(mysql_q)

    print("=" * 64)
    print("Milvus banking 孤儿/重复向量分析（只读）")
    print("=" * 64)
    print(f"MySQL banking question 数: {mysql_total}")
    print(f"Milvus banking 向量总数:   {mv_total}")
    print(f"  - 与 MySQL 重合: {in_both}")
    print(f"  - 孤儿(question 不在 MySQL banking): {len(orphan)}")
    print(f"  - 域内重复(同 question 多次插入超出首次): {dup_extra}")
    print(f"\n  可清理候选估算 = 孤儿 + 域内重复额外 = {len(orphan) + dup_extra}")
    print("\n孤儿抽样（前 10 条 question）:")
    for r in orphan[:10]:
        print(f"   id={r.get('id')}  {str(r.get('question',''))[:50]}")
    print("\n重复 question 抽样（前 5 个, 出现次数）:")
    for q, c in mv_counter.most_common(6):
        if c > 1:
            print(f"   x{c}: {str(q)[:50]}")

    out = f"logs/analyze_banking_orphans_{datetime.now():%Y%m%d_%H%M%S}.json"
    os.makedirs("logs", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mysql_banking": mysql_total, "milvus_banking": mv_total,
                   "in_both": in_both, "orphan": len(orphan), "dup_extra": dup_extra,
                   "orphan_sample": [{"id": r.get("id"), "q": r.get("question")} for r in orphan[:20]]},
                  f, ensure_ascii=False, indent=2)
    print(f"\n报告已写: {out}")


if __name__ == "__main__":
    main()
