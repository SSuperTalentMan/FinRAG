# -*- coding: utf-8 -*-
"""P1 优化复验：①few-shot 6 例真实出数；②caliber_check 双测（正确零误报/偏口径全抓）。"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.nl2sql import prompts, caliber_check
from rag_qa.nl2sql.schema import GeneratedSQL
from rag_qa.nl2sql import sqlguard, executor

print("=" * 60)
print("A) few-shot 示例执行级验证（真实 biz_demo）")
print("=" * 60)
raw = prompts.SQL_FEWSHOT
blocks = re.split(r"(?=示例\d+ )", raw)
ran = 0
ok = 0
for b in blocks:
    if "SELECT" not in b.upper():
        continue
    m = re.search(r"SELECT.*", b, re.S)
    sql = m.group(0).strip().rstrip(";").strip()
    head = b.strip().split("\n")[0][:40]
    ran += 1
    try:
        g = sqlguard.check(sql, {"fact_orders", "fact_order_items", "fact_refunds", "dim_store", "dim_product"})
        if not g.ok:
            print(f"  [{ran}] {head}  -> GUARD FAIL: {g.reject_reason}")
            continue
        cols, rows = asyncio.run(executor.run_readonly_sql(sql))
        ok += 1
        sample = rows[0] if rows else None
        print(f"  [{ran}] {head}  -> OK rows={len(rows)} sample={sample}")
    except Exception as e:  # noqa
        print(f"  [{ran}] {head}  -> EXEC ERR: {type(e).__name__}: {str(e)[:80]}")
print(f"few-shot 执行：{ok}/{ran} 成功")

print()
print("=" * 60)
print("B) caliber_check 双测：正确 SQL 应全 None")
print("=" * 60)
correct = [
    ("客单价 Top10 门店", "SELECT s.store_name, ROUND(SUM(o.pay_amount)/COUNT(DISTINCT o.order_id),2) AS arpu FROM fact_orders o JOIN dim_store s ON s.store_id=o.store_id WHERE o.pay_status='PAID' GROUP BY s.store_id ORDER BY arpu DESC LIMIT 10"),
    ("退货率最高的五个商品", "SELECT p.product_name, ROUND(r.refunded_orders/NULLIF(po.paid_orders,0),4) AS refund_rate FROM (SELECT i.product_id, COUNT(DISTINCT o.order_id) AS paid_orders FROM fact_orders o JOIN fact_order_items i ON i.order_id=o.order_id WHERE o.pay_status='PAID' GROUP BY i.product_id) po JOIN (SELECT i.product_id, COUNT(DISTINCT o.order_id) AS refunded_orders FROM fact_orders o JOIN fact_order_items i ON i.order_id=o.order_id JOIN fact_refunds rf ON rf.order_id=o.order_id WHERE o.pay_status='PAID' GROUP BY i.product_id) r ON r.product_id=po.product_id JOIN dim_product p ON p.product_id=po.product_id ORDER BY refund_rate DESC LIMIT 5"),
    ("本季度复购率", "SELECT ROUND(COUNT(DISTINCT CASE WHEN cnt>=2 THEN c_id END)/NULLIF(COUNT(DISTINCT c_id),0),4) FROM (SELECT o.customer_id AS c_id, COUNT(DISTINCT o.order_id) cnt FROM fact_orders o WHERE o.pay_status='PAID' AND QUARTER(o.order_date)=QUARTER(CURDATE()) AND YEAR(o.order_date)=YEAR(CURDATE()) GROUP BY o.customer_id) t"),
    ("今年 GMV 同比增长", "SELECT ROUND((SUM(CASE WHEN YEAR(o.order_date)=YEAR(CURDATE()) THEN o.pay_amount END)-SUM(CASE WHEN YEAR(o.order_date)=YEAR(CURDATE())-1 THEN o.pay_amount END))/NULLIF(SUM(CASE WHEN YEAR(o.order_date)=YEAR(CURDATE())-1 THEN o.pay_amount END),0),4) FROM fact_orders o WHERE o.pay_status='PAID'"),
]
allnone = True
for q, sql in correct:
    r = caliber_check.check_critical_caliber(GeneratedSQL(sql=sql), q)
    flag = "None" if r is None else f"误报! {r}"
    if r is not None:
        allnone = False
    print(f"  正确: {q[:18]:<20} -> {flag}")

print()
print("=" * 60)
print("C) caliber_check 双测：偏口径 SQL 应全抓出")
print("=" * 60)
biased = [
    ("客单价 Top10 门店", "SELECT s.store_name, AVG(o.pay_amount) AS arpu FROM fact_orders o JOIN dim_store s ON s.store_id=o.store_id WHERE o.pay_status='PAID' GROUP BY s.store_id ORDER BY arpu DESC LIMIT 10"),
    ("退货率最高的五个商品", "SELECT p.product_name, ROUND(SUM(rf.refund_amount)/NULLIF(SUM(o.pay_amount),0),4) AS refund_rate FROM fact_orders o JOIN fact_order_items i ON i.order_id=o.order_id JOIN dim_product p ON p.product_id=i.product_id LEFT JOIN fact_refunds rf ON rf.order_id=o.order_id WHERE o.pay_status='PAID' GROUP BY p.product_id ORDER BY refund_rate DESC LIMIT 5"),
    ("今年 GMV 同比增长", "SELECT ROUND((SUM(CASE WHEN YEAR(o.order_date)=YEAR(CURDATE()) THEN o.pay_amount END)-SUM(o.pay_amount))/NULLIF(SUM(o.pay_amount),0),4) FROM fact_orders o WHERE o.pay_status='PAID'"),
    ("各门店上月业绩完成率", "SELECT s.store_name, ROUND(SUM(o.pay_amount)/100000,4) AS completion_rate FROM fact_orders o JOIN dim_store s ON s.store_id=o.store_id GROUP BY s.store_id"),
]
allcatch = True
for q, sql in biased:
    r = caliber_check.check_critical_caliber(GeneratedSQL(sql=sql), q)
    flag = "抓出" if r else "漏抓!!"
    if r is None:
        allcatch = False
    print(f"  偏口径: {q[:18]:<20} -> {flag}: {r}")

print()
print("SUMMARY", "fewshot_ok=%d/%d" % (ok, ran),
      "caliber_correct_allNone=%s" % allnone,
      "caliber_biased_allCatch=%s" % allcatch)
