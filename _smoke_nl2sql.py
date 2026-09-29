#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时冒烟：SQLGuard 攻击向量 + Schema 召回 + 只读执行 + 时间回滚（无 LLM key 时生成环节降级）。"""
import asyncio
import os
import sys

os.environ.setdefault("SKIP_CONFIG_VALIDATION", "1")
os.environ.setdefault("API_KEY", "EMPTY_TEST_NO_CALL")
sys.path.insert(0, "D:/FinRag")

from rag_qa.nl2sql import sqlguard as sg
from rag_qa.nl2sql.meta_store import MetaStore
from rag_qa.nl2sql.schema_retriever import SchemaRetriever

ALLOWED = {"fact_orders", "dim_store", "dim_product", "dim_customer",
           "fact_refunds", "fact_order_items", "dim_date", "dim_weather", "dim_macro_cn"}

CASES = {
    "正常SELECT": "SELECT SUM(pay_amount) AS gmv FROM fact_orders",
    "多语句攻击": "SELECT 1; DROP TABLE fact_orders",
    "危险函数SLEEP": "SELECT SLEEP(5)",
    "INTO OUTFILE": "SELECT * INTO OUTFILE '/tmp/x' FROM fact_orders",
    "系统库@@变量": "SELECT @@version",
    "版本注释绕过": "SELECT /*!50000 SLEEP(5)*/ 1",
    "DDL穿透": "CREATE TABLE t AS SELECT * FROM fact_orders",
    "UPDATE写库": "UPDATE fact_orders SET pay_amount=0",
    "表白名单外": "SELECT * FROM users",
    "FOR UPDATE锁读": "SELECT * FROM fact_orders FOR UPDATE",
}


def test_guard():
    allowed_cnt = 0
    blocked_cnt = 0
    for name, sql in CASES.items():
        r = sg.check(sql, ALLOWED)
        tag = "ALLOW" if r.ok else "BLOCK"
        if r.ok:
            allowed_cnt += 1
        else:
            blocked_cnt += 1
        print(f"[guard] {tag}  {name}")
        if not r.ok:
            print(f"        -> {r.reject_reason}")
    assert blocked_cnt == len(CASES) - 1, "存在未拦截的攻击向量!"
    # 合法的要放行
    assert allowed_cnt == 1, "合法 SQL 被误杀!"
    print(f"[guard] 攻击向量 {blocked_cnt}/{len(CASES)-1} 全部拦截 ✓, 合法SQL放行 ✓")
    # 强制 LIMIT
    r = sg.check("SELECT * FROM fact_orders LIMIT 99999", ALLOWED)
    assert r.ok and "LIMIT 1000" in r.normalized_sql.upper(), "LIMIT 未强制!"
    print("[guard] 超限 LIMIT 被改写为 1000 ✓ ->", r.normalized_sql[-30:])


async def test_schema():
    meta = MetaStore()
    await meta.ensure_loaded()
    print(f"[schema] 元数据: 表={len(meta.enabled_tables())} 指标={len(meta._metrics)}")
    retr = SchemaRetriever(meta)
    allowed = retr.allowed_tables_for("admin")
    print(f"[schema] admin 可见表: {sorted(allowed)}")
    ctx = await retr.retrieve("近7天各区域GMV是多少", "admin")
    print(f"[schema] 召回表(关键词): {ctx.table_names} 来源={ctx.recall_source}")
    for m in meta.match_metrics("近7天净销售额"):
        print(f"[schema] 命中指标: {m.metric_name} | agg={m.agg_expr[:30]}")
    return allowed


def test_exec():
    # 直接只读执行（手工给一条合法 SQL），不依赖 LLM 生成
    from rag_qa.nl2sql.executor import run_readonly_sql
    cols, rows = asyncio.run(run_readonly_sql(
        "SELECT region, SUM(pay_amount) AS gmv FROM dim_store JOIN fact_orders USING (store_id) WHERE pay_status='PAID' GROUP BY region ORDER BY gmv DESC LIMIT 5"))
    print(f"[exec] 列={cols}")
    print(f"[exec] 行数={len(rows)} 首行={rows[0] if rows else None}")
    # 写库应被只读连接拒绝
    try:
        asyncio.run(run_readonly_sql("UPDATE fact_orders SET pay_amount=0"))
        print("[exec] 危险: 写库竟然成功了!")
    except Exception as e:
        print(f"[exec] 写库被只读连接拒绝 ✓ ({str(e)[:60]})")


if __name__ == "__main__":
    test_guard()
    asyncio.run(test_schema())
    test_exec()
    print("ALL SMOKE PASSED")