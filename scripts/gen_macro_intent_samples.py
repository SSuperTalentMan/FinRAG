#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/gen_macro_intent_samples.py — 生成宏观政策类意图分类训练样本

目的：gov.cn 入库的宏观政策文本多以「《通知》出台有什么背景？」等模糊措辞入库，
BERT 微调模型对「货币政策如何支持实体经济」「科创板注册制方向」等明确政策 Query
置信度低、易误路由到 investment_banking，导致对应 gov 数据在严格领域过滤下检索不到。
本脚本按 11 领域（侧重 gov 富集的 8 个）生成自然问句样本，供 train_intent_bert 增量训练。

输出：data/intent_macro_samples.jsonl，每行 {"text": 问句, "label": 领域}
"""
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "intent_macro_samples.jsonl"

# 与 rag_qa/core/query_classifier.py 的 LABEL_MAP 对齐
LABEL = {
    "banking": 0, "corporate_finance": 1, "financial_accounting": 2, "general": 3,
    "financial_markets": 4, "fintech": 5, "insurance": 6, "investment_banking": 7,
    "personal_finance": 8, "risk_management": 9, "stock_market": 10,
}

TERMS = {
    "financial_markets": [
        "货币政策", "降准", "降息", "存款准备金率", "人民币汇率", "债券市场", "国债",
        "地方政府专项债", "资本市场改革", "科创板", "股票发行注册制", "北交所",
        "宏观审慎政策", "市场流动性", "通货膨胀", "金融对外开放", "外汇储备",
        "货币市场", "直接融资", "间接融资",
    ],
    "stock_market": [
        "股票市场", "A股", "上市公司现金分红", "股票减持", "退市机制", "IPO",
        "再融资", "北向资金", "价值投资", "指数", "换手率", "市盈率",
        "市值管理", "投资者保护", "股价异常波动",
    ],
    "risk_management": [
        "地方政府债务风险", "隐性债务", "化债", "城投债", "金融风险防范化解",
        "系统性金融风险", "影子银行", "非法集资", "监管套利", "压力测试",
        "地方债务率", "房地产金融风险",
    ],
    "corporate_finance": [
        "企业融资成本", "民营企业融资", "中小企业融资", "减税降费", "营商环境",
        "企业并购重组", "国企改革", "混合所有制改革", "股权融资", "企业估值",
        "专精特新企业", "惠企政策",
    ],
    "personal_finance": [
        "个人养老金", "养老第三支柱", "居民储蓄", "个人理财", "个人所得税专项附加扣除",
        "居民消费", "住房公积金", "财富管理", "个税汇算", "居民可支配收入",
    ],
    "banking": [
        "商业银行资本充足率", "普惠小微贷款", "存款利率市场化", "不良贷款率",
        "政策性金融", "村镇银行", "数字人民币", "征信体系", "房贷利率",
        "商业银行资本管理办法", "中小银行改革",
    ],
    "insurance": [
        "个人养老金保险", "长期护理保险", "农业保险", "大病保险", "医保改革",
        "养老保险全国统筹", "商业健康险", "巨灾保险", "保险资金运用", "车险综改",
    ],
    "fintech": [
        "数字人民币", "金融科技", "移动支付", "跨境支付", "区块链在金融应用",
        "监管科技", "数据要素", "人工智能金融监管", "数字货币", "开放银行",
    ],
    # 以下三个 gov 数据较少，少量补充以平衡
    "investment_banking": [
        "基础设施REITs", "资产证券化", "并购重组", "投资银行承销", "私募股权",
        "政府引导基金",
    ],
    "financial_accounting": [
        "政府会计制度", "企业会计准则", "财务信息披露", "审计准则", "预算绩效管理",
    ],
    "general": [
        "今天天气怎么样", "怎么学习金融知识", "推荐一本经济学书", "什么是GDP",
        "如何做好职业规划", "请解释一下通货膨胀通俗说法",
    ],
}

TEMPLATES = [
    "{t}是什么？",
    "{t}是怎么规定的？",
    "{t}对实体经济有什么影响？",
    "我国{t}的现状如何？",
    "如何理解和把握{t}？",
    "{t}政策有哪些最新变化？",
    "关于{t}，监管部门是怎么说的？",
    "{t}和普通老百姓有什么关系？",
    "近年来{t}有哪些重要举措？",
    "{t}未来会朝着什么方向发展？",
    "请介绍一下{t}的主要政策安排。",
    "{t}和{t2}之间有什么联系？",
    "为什么国家要重视{t}？",
    "{t}的实施效果怎么样？",
    "权威部门对{t}是怎么回应的？",
]


def gen():
    random.seed(7)
    samples = []
    for domain, terms in TERMS.items():
        n = 0
        attempts = 0
        while n < 150 and attempts < 4000:
            attempts += 1
            t = random.choice(terms)
            tpl = random.choice(TEMPLATES)
            if "{t2}" in tpl:
                t2 = random.choice(terms)
                if t2 == t:
                    continue
                text = tpl.format(t=t, t2=t2)
            else:
                text = tpl.format(t=t)
            samples.append({"text": text, "label": LABEL[domain]})
            n += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"生成 {len(samples)} 条 -> {OUT}")


if __name__ == "__main__":
    gen()
