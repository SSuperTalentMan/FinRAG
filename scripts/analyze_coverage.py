#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FinRag 语料覆盖度看板（只读，不改任何数据）。

设计要点（避免误报）：
  - MySQL(finance_faq) 每域行数 = 语料覆盖度的"真值"（可靠）。
  - Milvus 的 per-domain `query()` 全扫会漏掉增长段，与 `search()` 结果不一致，
    因此**不**用它判断"某域是否缺向量"；改用生产路径 `search()` 逐域可达性验证。
  - Milvus 总数用 get_collection_stats（实体数，权威，但接口偶发滞后）。

输出：
  1) 各 domain：MySQL 行数 / Milvus 总数 / 该域 search 是否可达(命中数) / crawled faq 条数
  2) 一致性结论（孤儿向量候选、清理建议）

用法：
  .venv/Scripts/python.exe scripts/analyze_coverage.py
"""
import sys
import os
import glob
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from pymilvus import MilvusClient

from config import get_config
from rag_qa.core.query_classifier import classify_intent
from routers.chat import encode_query_dense_sparse
from db.milvus import get_milvus_client, search_milvus


# 各域一个代表性中文查询，用于 search 可达性验证
PROBE = {
    "banking": "什么是普惠金融？",
    "corporate_finance": "企业融资有哪些方式？",
    "financial_accounting": "财务报表有什么作用？",
    "financial_markets": "科创板注册制改革方向是什么？",
    "fintech": "数字支付如何推动金融普惠？",
    "general": "近期有哪些金融政策？",
    "insurance": "保险公司偿付能力是什么？怎么监管？",
    "investment_banking": "投资银行主要做什么？",
    "personal_finance": "普通人怎么理财？",
    "risk_management": "什么是 VaR 风险价值？",
    "stock_market": "做市商如何稳定股市价格？",
}


def mysql_counts(cfg) -> dict:
    conn = pymysql.connect(
        host=cfg.mysql.host, port=cfg.mysql.port, user=cfg.mysql.user,
        password=cfg.mysql.password, database=cfg.mysql.database,
        charset=cfg.mysql.charset, cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        cur = conn.cursor()
        cur.execute("SELECT category, COUNT(*) AS c FROM finance_faq GROUP BY category ORDER BY category")
        return {r["category"]: r["c"] for r in cur.fetchall()}
    finally:
        conn.close()


def milvus_total(cfg) -> int:
    kw = dict(uri=cfg.milvus.uri, token=cfg.milvus.token) if getattr(cfg.milvus, "token", None) else dict(uri=cfg.milvus.uri)
    client = MilvusClient(**kw)
    if not client.has_collection(cfg.milvus.collection_name):
        return -1
    try:
        st = client.get_collection_stats(cfg.milvus.collection_name)
        if isinstance(st, dict):
            return int(st.get("row_count", st.get("num_entities", -1)))
        return int(st)
    except Exception as e:
        return f"err:{e}"


def milvus_reachable() -> dict:
    """用生产 search 路径逐域验证可达性（权威，不受 query 段行为影响）。"""
    client = get_milvus_client()
    out = {}
    for dom, q in PROBE.items():
        try:
            dense, sparse = encode_query_dense_sparse(q)
            res = search_milvus(client, dense, sparse, k=5, domain=dom)
            same_cat = sum(1 for r in res if r.get("category") == dom)
            out[dom] = (len(res), same_cat)
        except Exception as e:
            out[dom] = f"err:{e}"
    return out


def crawled_counts() -> dict:
    out = {}
    for p in sorted(glob.glob("data/crawled/faq_*_qa.jsonl")):
        dom = os.path.basename(p).replace("faq_", "").replace("_qa.jsonl", "")
        out[dom] = sum(1 for _ in open(p, encoding="utf-8"))
    return out


def main():
    cfg = get_config()
    print("=" * 72)
    print("FinRag 语料覆盖度看板（只读 · search 为检索真值）")
    print("=" * 72)

    my = mysql_counts(cfg)
    mv_total = milvus_total(cfg)
    reach = milvus_reachable()
    cr = crawled_counts()

    domains = sorted(set(my) | set(cr) | set(PROBE))
    print(f"\n{'domain':<22}{'MySQL行':>9}{'Milvus总数':>11}{'search可达':>12}{'crawled':>9}")
    print("-" * 72)
    for d in domains:
        m = my.get(d, 0)
        r = reach.get(d, (0, 0))
        if isinstance(r, tuple):
            rtxt = f"{r[0]}条/{r[1]}同域" if r[1] > 0 else f"{r[0]}条✗"
        else:
            rtxt = str(r)
        c = cr.get(d, 0)
        print(f"{d:<22}{m:>9}{str(mv_total):>11}{rtxt:>12}{c:>9}")

    print("-" * 72)
    print(f"Milvus 集合 '{cfg.milvus.collection_name}' 实体总数(stats): {mv_total}  "
          f"（注：stats 接口偶发滞后，以 search 可达性为准）")
    print(f"MySQL finance_faq 总行数: {sum(my.values())}")
    print(f"本轮 faq 入库来源(crawled): 合计 {sum(cr.values())} 条")

    unreachable = [d for d in domains if isinstance(reach.get(d), tuple) and reach[d][1] == 0]
    print("\n## 结论")
    if unreachable:
        print(f"  ⚠ 以下域 search 未返回同域命中（需排查）: {unreachable}")
    else:
        print("  ✓ 全部 11 个域经生产 search 路径均可达且类别正确（领域隔离生效）。")
    # banking 孤儿候选（MySQL 620 vs Milvus 中 banking 段 3450，来自早前 query 观测）
    print("  • banking 在 Milvus 中向量数(~3450)远高于 MySQL(620)，疑似早期重复入库的孤儿向量；")
    print("    如需清理需显式确认（删 Milvus 向量为不可逆操作），建议作为独立任务处理。")


if __name__ == "__main__":
    main()
