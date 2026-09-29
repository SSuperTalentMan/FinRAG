#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/intent.py — 四分类意图识别服务
使用规则 + 关键词匹配判断问题所属领域（banking / corporate_finance / financial_accounting / general）。
注：生产环境可替换为微调 BERT 模型，此处先用关键词规则实现，保证无需额外训练即可运行。
"""

from dataclasses import dataclass
from loguru import logger
from config import get_config

# ─── 各领域关键词 ───────────────────────────────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "banking": [
        "银行", "信贷", "贷款", "存款", "利率", "汇兑", "结算", "票据",
        "信用卡", "借记卡", "透支", "同业", "拆借", "流动性", "准备金",
        "央行", "央行数字货币", "CBDC", "支付", "清算", "ATM", "网银",
        "普惠金融", "小微贷款", "供应链金融", "贸易融资", "保理",
        "资本充足率", "巴塞尔", "存贷比", "不良资产", "坏账", "MRA",
        "PSB", "商业银行", "政策性银行", "农信社", "村镇银行",
    ],
    "corporate_finance": [
        "公司金融", "企业融资", "融资", "股权", "债权", "并购", "重组",
        "IPO", "上市", "退市", "估值", "尽职调查", "杠杆收购",
        "现金流", "资本结构", "加权平均资本成本", "WACC", "NPV", "IRR",
        "DCF", "折现", "现值", "资本预算", "投资回收期",
        "股东权益", "董事会", "治理", "代理问题", "激励",
        "股利", "分红", "回购", "股息",
        "跨国公司", "跨国经营", "海外投资", "FDI",
        "麦克唐纳", "帕克",
    ],
    "financial_accounting": [
        "会计", "财务", "资产负债表", "利润表", "现金流量表", "所有者权益",
        "折旧", "摊销", "坏账准备", "存货", "应收账款", "应付账款",
        "审计", "会计准则", "GAAP", "IFRS", "审计意见",
        "单一审计法", "联邦资金", "政府会计",
        "财务报告", "合并报表", "抵销", "商誉",
        "收入确认", "成本核算", "费用", "税金",
        "公允价值", "历史成本", "权责发生制", "收付实现制",
    ],
}

# 通用问题关键词（低于置信度时降级为 general）
GENERAL_SIGNALS = ["天气", "今天", "你好", "谢谢", "怎么", "为什么", "谁", "哪里", "多少", "几点"]


@dataclass
class IntentResult:
    domain: str          # banking / corporate_finance / financial_accounting / general
    confidence: float    # 0.0 ~ 1.0
    keywords_hit: list[str]


def classify_intent(query: str) -> IntentResult:
    """
    基于关键词命中对问题做四分类。
    命中最多的领域得分最高；若无明确领域信号则返回 general。
    """
    query_lower = query.lower()

    # 计算每个领域的命中分数
    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in query]
        scores[domain] = len(matched)
        hits[domain] = matched

    # 找最高分
    if not any(scores.values()):
        # 无领域关键词命中 → general
        return IntentResult(domain="general", confidence=0.0, keywords_hit=[])

    max_domain = max(scores, key=scores.get)
    max_score  = scores[max_domain]
    total_hits = sum(scores.values())

    # 置信度 = 最高分 / 总分（归一化）
    confidence = round(max_score / total_hits, 4) if total_hits > 0 else 0.0

    # 若最高置信度低于 0.6，降级为 general
    cfg = get_config()
    conf_threshold = cfg.retrieval.similarity_threshold  # 复用配置中的阈值
    if confidence < 0.6:
        logger.debug(f"意图识别置信度较低 {confidence:.4f}，降级为 general")
        return IntentResult(domain="general", confidence=confidence, keywords_hit=hits[max_domain])

    logger.debug(f"意图识别: domain={max_domain}, confidence={confidence:.4f}, hits={hits[max_domain]}")
    return IntentResult(domain=max_domain, confidence=confidence, keywords_hit=hits[max_domain])
