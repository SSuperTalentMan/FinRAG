#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/caliber_check.py — 派生指标口径确定性校验（post-check）。

问数质量的最大短板是「派生指标口径」：客单价被算成 AVG(pay_amount)、退货率被算成
金额比/漏联退款表、同比环比混淆、季度类漏 QUARTER/YEAR 过滤等。这类错误 LLM 判分
才发现、且偶发，难以稳定治理。这里用正则抽取生成 SQL 的口径特征做**确定性**回检，
命中危险模式就返回结构化反馈，交由 service 回喂模型重写（不依赖 LLM 二次判分）。

仅做"明显会偏离业务口径"的硬约束检查，不取代语义理解；漏报（软口径偏差）仍由评估
脚本的 LLM 判分覆盖。
"""
from __future__ import annotations

import re

from rag_qa.nl2sql.schema import GeneratedSQL


def check_critical_caliber(gen: GeneratedSQL, question: str) -> str | None:
    """返回 None 表示口径无明显问题；否则返回回喂模型的修正提示（中文）。

    仅当问题命中对应指标词且 SQL 表现出危险写法时才报警，避免误伤正常查询。
    """
    sql = (gen.sql or "").strip()
    if not sql:
        return None
    q = question or ""
    low = sql.lower()
    hints: list[str] = []

    # 1) 客单价：必须按"已支付订单数"除，即含 COUNT(DISTINCT ... order_id)；
    #    出现 AVG(pay_amount) 也算偏口径（客单价是订单均额不是行均额）
    if "客单价" in q:
        if re.search(r"avg\s*\(\s*\w*\.?pay_amount", low):
            hints.append("客单价口径错误：禁止用 AVG(pay_amount)，客单价=SUM(pay_amount)/COUNT(DISTINCT 已支付订单order_id)")
        elif not re.search(r"count\s*\(\s*distinct\s+\w*\.?order_id", low):
            hints.append("客单价口径错误：必须用 SUM(pay_amount)/COUNT(DISTINCT 已支付订单order_id) 作分母")

    # 2) 退货率：必须联 fact_refunds 表（用表名判断，避免别名 refund_rate 误判），
    #    且按"订单数"占比（非金额）
    if "退货率" in q:
        if "fact_refunds" not in low:
            hints.append("退货率口径错误：必须 JOIN fact_refunds 关联退款记录，按订单数占比计算")
        if re.search(r"sum\s*\(\s*\w*\.?refund_amount", low):
            hints.append("退货率口径错误：退货率=发生退货的订单数/已支付订单数（按订单数），"
                         "禁止用 SUM(refund_amount) 算金额比")

    # 3) 季度类：本季度/上季度等必须带 QUARTER()/YEAR() 过滤，不能只按月份
    if re.search(r"季度", q):
        if "quarter(" not in low:
            hints.append("时间口径错误：问『季度』必须用 QUARTER(o.order_date)=QUARTER(CURDATE()) "
                         "AND YEAR(o.order_date)=YEAR(CURDATE()) 限定本季度，禁止只按月份过滤")

    # 4) 同比/环比：必须出现两年/两月对比（DATE_SUB 或 YEAR/MONTH 算术均算合法）
    if "同比" in q and "环比" not in q:
        if "interval 1 year" not in low and not re.search(
            r"year\s*\([\w.()]+\s*\)\s*=\s*year\s*\([\w.()]+\s*\)\s*-\s*1", low
        ):
            hints.append("同比口径错误：同比=本年本月 vs 去年同月，需用 DATE_SUB(...,INTERVAL 1 YEAR) "
                         "或 YEAR(order_date)=YEAR(CURDATE())-1 取去年同月对比")
    if "环比" in q and "同比" not in q:
        if low.count("date_sub") < 2 and not re.search(r"month\s*\([\w.()]+\s*\)\s*-\s*1", low):
            hints.append("环比口径错误：环比=上月 vs 上上月，需两个 DATE_SUB(INTERVAL 1/2 MONTH) "
                         "或 MONTH(...) 与 MONTH(...)-1 取相邻两月对比")

    # 5) 完成率/达成率：biz_demo 没有目标值/配额/考核表，任何"算出来"的完成率都是编造，
    #    必须 REFUSE（与 prompts.py 规则7/示例6 一致）
    if "完成率" in q or "达成率" in q:
        hints.append("业绩完成率/达成率需要目标值或考核表，biz_demo 无此表，禁止编造 SQL，"
                     "必须输出 REFUSE")

    if not hints:
        return None
    return "；".join(hints)
