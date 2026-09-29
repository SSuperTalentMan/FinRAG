#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FinRag 检索质量回归护栏（只读，不改任何数据）。

固化两类检查，作为后续每次改动的回归基线：
  1) 6 个核心问（来自 e2e_verify_qa.QUESTIONS）：跑生产路径，报告意图/Top1 分数/弱命中。
  2) 10 个域 faq 探针：用代表性中文问法，验证该域在 Milvus 经 search 可达且类别正确。

判定口径：
  - 弱命中：Top1 score < 0.7（与端到端验收一致）。
  - 域错配：检索 Top1 真实 category != 期望域（需 search 透出 category）。

用法：
  .venv/Scripts/python.exe scripts/eval_coverage.py            # 仅检索层（快）
  .venv/Scripts/python.exe scripts/eval_coverage.py --answer   # 额外生成 LLM 答案（慢/耗 token）
输出：表格 + logs/eval_coverage_<ts>.json
"""
import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_qa.core.query_classifier import classify_intent
from routers.chat import _build_context, generate_answer


# 6 个核心问（与 e2e_verify_qa.QUESTIONS 保持一致）
CORE_QUESTIONS = [
    "货币政策如何支持实体经济",
    "个人养老金怎么参加",
    "科创板注册制改革方向是什么",
    "上市公司分红有哪些规定",
    "地方政府债务风险如何防范",
    "什么是普惠金融",
]

# 10 域 faq 探针（代表性中文问法）
DOMAIN_PROBES = {
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


def check_core(answer: bool):
    rows = []
    for q in CORE_QUESTIONS:
        intent = classify_intent(q)
        ctx, sources = _build_context(q, intent)
        top = sources[0] if sources else {}
        score = top.get("score", 0.0) or 0.0
        rows.append({
            "type": "core", "question": q, "domain": intent.domain,
            "conf": round(intent.confidence, 4),
            "top1_score": round(score, 4), "top1_source": top.get("source", ""),
            "weak": score < 0.7,
        })
        if answer:
            try:
                ans = generate_answer(q, ctx, intent.domain)
                rows[-1]["answer_len"] = len(ans)
            except Exception as e:
                rows[-1]["answer_err"] = str(e)[:80]
    return rows


def check_domains():
    rows = []
    for dom, q in DOMAIN_PROBES.items():
        intent = classify_intent(q)
        ctx, sources = _build_context(q, intent)
        top = sources[0] if sources else {}
        score = top.get("score", 0.0) or 0.0
        # _build_context 透出的 source 不含 category；此处仅记录可达性与分数
        rows.append({
            "type": "domain", "domain": dom, "probe": q,
            "routed": intent.domain, "routed_ok": intent.domain == dom,
            "top1_score": round(score, 4), "top1_source": top.get("source", ""),
            "reachable": len(sources) > 0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", action="store_true", help="额外生成 LLM 答案（慢/耗 token）")
    args = ap.parse_args()

    core = check_core(args.answer)
    dom = check_domains()
    allrows = core + dom

    weak = [r for r in core if r["weak"]]
    unreachable = [r for r in dom if not r["reachable"]]
    misrouted = [r for r in dom if not r["routed_ok"]]

    print("=" * 72)
    print("FinRag 检索质量回归护栏")
    print("=" * 72)
    print("\n【核心 6 问】")
    print(f"{'问题':<22}{'域':<18}{'conf':>6}{'Top1':>8}  状态")
    for r in core:
        print(f"{r['question'][:20]:<22}{r['domain']:<18}{r['conf']:>6}{r['top1_score']:>8}  {'弱命中✗' if r['weak'] else '强✓'}")

    print("\n【10 域 faq 探针】")
    print(f"{'域':<20}{'路由':<18}{'Top1':>8}  可达")
    for r in dom:
        print(f"{r['domain']:<20}{r['routed']:<18}{r['top1_score']:>8}  {'✓' if r['reachable'] else '✗'}")

    print("\n【汇总】")
    print(f"  核心问弱命中: {len(weak)}  | 域探针不可达: {len(unreachable)}  | 域探针误路由: {len(misrouted)}")
    if weak:
        print("  弱命中清单:", [r["question"] for r in weak])
    if unreachable:
        print("  不可达域:", [r["domain"] for r in unreachable])
    if misrouted:
        print("  误路由:", [(r["domain"], r["routed"]) for r in misrouted])

    out = f"logs/eval_coverage_{datetime.now():%Y%m%d_%H%M%S}.json"
    os.makedirs("logs", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"core": core, "domains": dom, "weak": len(weak),
                   "unreachable": len(unreachable), "misrouted": len(misrouted)}, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已写: {out}")


if __name__ == "__main__":
    main()
