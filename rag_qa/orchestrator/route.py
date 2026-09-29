#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/orchestrator/route.py — 意图 -> Skill 路由（规则先行 + 业务实体证据精修 + 可选 LLM 兜底）。

两阶段设计
----------
阶段一（纯规则，离线可复现）：评测脚本 scripts/eval_routing.py 只依赖 `_decide_skill`。
    review > multimodal > rag(显式知识词) > nl2sql > chitchat > rag(默认兜底)

阶段二（数据驱动，仅运行时）：用 nl2sql 的 MetaStore（table_registry / column_registry /
    metric_dict）构建「业务实体词表」，对问题做加权打分 —— 只有在业务库里真实存在的
    表/字段/指标/枚举值才算证据。据此做两件词表做不到的事：
      1. 规则判 nl2sql 但没有任何业务实体命中 -> 降级 rag（"存款保险最高赔付多少"
         这类知识问题被泛词误判成问数的最后一道防线）；
      2. 规则走默认兜底 rag，但业务实体强命中且带聚合意图 -> 升级 nl2sql
         （"华东大区上个月单量多少"这类没踩中任何关键词的问数）。

为什么必须双阶段：纯词表是黑名单博弈，业务表一变（新表/新指标）就漏，且泛词
（最高/平均/本月/区域）在知识问句里同样高频。把"业务库里到底有没有这个实体"作为
判据，才能从根上消除误判；规则保留为快速通道与降级兜底（元数据不可用时完全退回旧行为）。

review 优先级最高的原因：审查请求往往同时含"图片/扫描件"信号（"帮我审这份合同的
扫描件"），若按旧顺序先判 multimodal 会只给上传引导，丢掉真正的审查意图。
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from config import get_config

logger = logging.getLogger(__name__)

SKILLS = ("rag", "nl2sql", "chitchat", "multimodal", "review")

# 结构化问数强信号词：仅保留"业务口径/经营分析"强相关词。
# 刻意剔除"最高/最低/平均/增长/下降/本月/今年/指标/区域/商品"等泛词——
# 它们在金融知识问句里同样高频（如"存款保险最高赔付多少"），保留会把知识问题误判成问数。
_NL2SQL_STRONG = (
    "GMV", "gmv", "销售额", "销售金额", "销售总额", "营收", "营业收入",
    "订单量", "订单数", "单量", "销量", "销售量", "退货率", "退款额",
    "客单价", "毛利率", "利润率", "净利润", "同比", "环比", "增速", "增长率",
    "排行", "榜单", "TOP", "top", "大区", "门店", "复购", "库存量",
    "交易额", "交易量", "笔数", "业绩", "完成率", "达成率",
    "总GMV", "京东", "近7天", "退款金额", "毛利额", "库存",
)

# 金融知识/政策/合规强信号词：命中即判 rag，优先于问数。
# 这些是本系统的知识库主体（监管政策、会计准则、业务释义），不可能出现在业务库表里。
_RAG_STRONG = (
    "存款保险", "征信", "信用报告", "条例", "办法", "规定", "细则", "指引",
    "监管", "合规", "反洗钱", "巴塞尔", "准备金", "会计准则", "折旧", "摊销",
    "IPO", "注册制", "上市", "招股", "披露", "停牌", "退市", "涨跌幅",
    "基金", "净值", "申购", "赎回", "定投", "保单", "理赔", "保费", "保险责任",
    "贷款", "房贷", "汇率", "LPR", "贴现", "承兑", "汇票", "信用证",
    "数字人民币", "洗钱", "非法集资", "诈骗", "投资者保护", "适当性",
    "政策", "法规", "法律", "司法解释", "通知", "公告", "征求意见稿",
    "什么是", "如何计算", "怎么算", "依据", "释义", "定义",
)

# 多模态/读图强信号词：需要解析图片或扫描件才能回答
_MULTIMODAL_STRONG = (
    "扫描", "扫描件", "图片", "图像", "图表", "截图", "拍照", "照片", "发票",
    "识别图中", "识别图片", "提取图中", "ocr", "OCR", "盖章", "红章", "条形码",
    "二维码", "表格图片", "手写", "pdf图片", "扫描照片", "上面写着", "图中显示",
    "化验单", "证件照", "单据", "票据",
)

# ── 文档审查（融合 DocAudit）：审查意图必须优先于 multimodal ──────────────────
# 强信号：单独出现即可判定（多为"审查/风险"类动词或直接点名合同）
_REVIEW_STRONG = (
    "审查", "审一下", "审这份", "审这个", "帮我审", "审合同", "审协议",
    "合规审查", "合规检查", "合规体检", "条款审查", "合同条款", "协议条款",
    "风险点", "有没有风险", "有风险吗", "违规条款", "踩红线", "不合规",
    "挑毛病", "查一下风险",
)
# 文档指示词 + 动作词的组合判定：降低"是否符合"之类泛词单独触发的误判
_REVIEW_DOC_HINT = (
    "这份", "这个合同", "这份合同", "本协议", "该合同", "条款", "正文",
    "合同", "协议", "文档", "材料", "文件", "章程", "招股书", "说明书",
)
_REVIEW_ACTION = (
    "审查", "审一下", "检查", "核对", "校验", "看看有没有", "有没有问题",
    "是否合规", "合规吗", "符合吗", "有问题吗", "哪里有问题",
    # 精确短语，避免裸词"风险"把"理财产品风险分级"等知识检索问题误带进审查
    "风险点", "有风险吗", "风险吗", "有什么风险", "有哪些风险", "风险在哪",
    # 合规判定短语：这些几乎只出现在"合同/协议是否合规"语境，配合文档指示词即判审查
    "符合法律", "符合法规", "符合监管", "符合规定",
)

# 寒暄/问候：仅在无业务词时路由到 chitchat
_CHITCHAT = (
    "你好", "您好", "嗨", "哈喽", "hello", "hi", "早上好", "中午好", "晚上好",
    "晚安", "再见", "拜拜", "谢谢", "感谢", "你是谁", "你能做什么", "你会什么",
    "都会什么", "能做什么", "会做什么",
    "在吗", "辛苦了", "怎么称呼", "自我介绍", "自我介绍一下",
    "做什么", "干什么", "能干嘛", "可以帮你", "帮忙吗", "哈喽你们好",
    "打个招呼", "打招呼", "聊天", "聊会", "周末", "愉快", "很高兴", "认识你",
    "哪些帮助", "轻松",
)

# 聚合/查询意图词：仅在「业务实体证据足够」时才把默认兜底升级为 nl2sql，
# 单独出现不具判据力（"存款保险最高赔付多少"也含"多少"）。
_AGG_INTENT = (
    "多少", "几个", "统计", "查询", "查一下", "汇总", "合计", "总计", "总共",
    "排名", "排行", "趋势", "分布", "对比", "相比", "分组", "按月", "按日",
    "每月", "每日", "各月", "各地", "各省", "各类", "清单", "明细", "列表",
)


def _has_any(q: str, words) -> bool:
    ql = q.lower()
    return any(w.lower() in ql for w in words)


# ══════════════════ 业务实体证据（数据驱动路由的核心）══════════════════════════
# 权重的含义：命中"指标口径"几乎必然是问数；命中"枚举样本值"（如城市名）次之；
# 命中字段说明里的片段最弱，只作辅助。
_W_METRIC = 3.0     # metric_dict：metric_name / synonyms
_W_TABLE = 2.5      # table_registry：table_name / display_name
_W_COLUMN = 1.5     # column_registry：字段名与其说明片段
_W_SAMPLE = 1.0     # column_registry：sample_values 枚举值

# 把中文说明切成语素：description 形如"订单金额（元，含税）"，整句匹配无意义
_SPLIT_SEG = re.compile(r"[、，,;；/|（(）)【】\[\]\s]+")
_TERM_MIN_LEN = 2
_TERM_MAX_LEN = 12

_ROUTE_META = None          # MetaStore 单例（与 NL2SQL 服务共享，避免重复加载元数据）
_TERMS: list[tuple[str, float]] | None = None
_TERMS_TS: float = 0.0
_TERMS_TTL: float = 300.0   # 元数据可能新增表/指标，词表定期重建


def get_route_meta():
    """返回共享的 MetaStore 实例（不在此处连库，避免阻塞事件循环）。"""
    global _ROUTE_META
    if _ROUTE_META is None:
        from rag_qa.nl2sql.meta_store import MetaStore

        _ROUTE_META = MetaStore()
    return _ROUTE_META


async def aensure_route_meta():
    """确保业务元数据已加载；失败仅告警（路由退回纯规则，不影响服务可用性）。"""
    meta = get_route_meta()
    try:
        await meta.ensure_loaded()
    except Exception as e:  # noqa: BLE001
        logger.warning("业务元数据加载失败，路由退回纯规则: %s", str(e)[:120])
    return meta


def reset_entity_cache() -> None:
    """元数据变更后调用，强制下次重建实体词表（测试与热更新用）。"""
    global _TERMS, _TERMS_TS
    _TERMS = None
    _TERMS_TS = 0.0


def _segments(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT_SEG.split(text or "")
            if _TERM_MIN_LEN <= len(s.strip()) <= _TERM_MAX_LEN]


def build_entity_terms(meta) -> list[tuple[str, float]]:
    """从业务元数据抽取可作判据的实体词（去重、按权重）。"""
    terms: list[tuple[str, float]] = []
    if meta is None:
        return terms

    def add(word, w):
        word = (word or "").strip()
        if len(word) < _TERM_MIN_LEN:
            return
        terms.append((word, w))

    for m in getattr(meta, "_metrics", []) or []:
        add(m.get("metric_name"), _W_METRIC)
        for syn in (m.get("synonyms") or "").split(","):
            add(syn, _W_METRIC)

    for t in getattr(meta, "table_registry_rows", []) or []:
        add(t.get("table_name"), _W_TABLE)
        add(t.get("display_name"), _W_TABLE)
        for seg in _segments(t.get("description")):
            add(seg, _W_TABLE * 0.6)

    for c in (meta.all_columns() if hasattr(meta, "all_columns") else []):
        add(c.get("column_name"), _W_COLUMN * 0.8)
        for seg in _segments(c.get("description")):
            add(seg, _W_COLUMN)
        for seg in _segments(c.get("sample_values")):
            add(seg, _W_SAMPLE)

    # 去重：同一词保留最高权重；长词优先在打分阶段屏蔽子串
    best: dict[str, float] = {}
    for w, s in terms:
        key = w.lower()
        if best.get(key, 0.0) < s:
            best[key] = s
    return [(w, s) for w, s in best.items()]


def _entity_terms(meta):
    global _TERMS, _TERMS_TS
    now = time.monotonic()
    if _TERMS is not None and now - _TERMS_TS < _TERMS_TTL:
        return _TERMS
    _TERMS = build_entity_terms(meta)
    _TERMS_TS = now
    return _TERMS


def score_biz_entities(question: str, meta=None) -> tuple[float, list[str]]:
    """业务实体加权打分：命中长词后屏蔽其子串，避免"销售额/净销售额"重复计分。

    返回 (score, hits)。score 为 0 表示问题里没有任何业务库实体 —— 这种情况
    即使踩中了问数泛词，也不该去查库。
    """
    if meta is None or not getattr(meta, "_loaded", False):
        return 0.0, []
    q = (question or "").lower()
    if not q:
        return 0.0, []
    hits: list[str] = []
    score = 0.0
    remaining = q
    # 长词优先：先让"净销售额"吃掉字符，剩下的才轮到"销售额"
    for term, weight in sorted(_entity_terms(meta), key=lambda x: -len(x[0])):
        tl = term.lower()
        if len(tl) < _TERM_MIN_LEN or tl not in remaining:
            continue
        hits.append(term)
        score += weight
        remaining = remaining.replace(tl, " ")
        if len(hits) >= 12:  # 防止超长问题刷分
            break
    return round(score, 2), hits


# ══════════════════ 规则路由（离线、确定性）═══════════════════════════════════
def _decide_skill_detail(question: str) -> tuple[str, bool]:
    """返回 (skill, explicit)。explicit=False 表示落到默认兜底 rag（未被任何规则命中）。"""
    q = (question or "").strip()
    if not q:
        return "rag", False
    if _has_any(q, _REVIEW_STRONG):
        return "review", True
    if (_has_any(q, _REVIEW_DOC_HINT) and _has_any(q, _REVIEW_ACTION)):
        return "review", True
    if _has_any(q, _MULTIMODAL_STRONG):
        return "multimodal", True
    if _has_any(q, _RAG_STRONG):
        return "rag", True
    if _has_any(q, _NL2SQL_STRONG):
        return "nl2sql", True
    if _has_any(q, _CHITCHAT):
        return "chitchat", True
    return "rag", False


def _decide_skill(question: str) -> str:
    """纯规则路由（供评测脚本离线复现）：review > multimodal > rag > nl2sql > chitchat。"""
    return _decide_skill_detail(question)[0]


def decide_skill_keywords(question: str) -> tuple[str, list[str]]:
    """返回 (skill, 命中的关键词组)。供评估/诊断展示路由依据（纯规则，不连库）。"""
    q = (question or "").strip()
    if not q:
        return "rag", []
    hits: dict[str, list[str]] = {
        "multimodal": [], "rag": [], "nl2sql": [], "chitchat": [], "review": [],
    }
    ql = q.lower()
    for w in _REVIEW_STRONG:
        if w.lower() in ql:
            hits["review"].append(w)
    for w in _MULTIMODAL_STRONG:
        if w.lower() in ql:
            hits["multimodal"].append(w)
    for w in _RAG_STRONG:
        if w.lower() in ql:
            hits["rag"].append(w)
    for w in _NL2SQL_STRONG:
        if w.lower() in ql:
            hits["nl2sql"].append(w)
    for w in _CHITCHAT:
        if w.lower() in ql:
            hits["chitchat"].append(w)
    if not hits["review"] and _has_any(q, _REVIEW_DOC_HINT):
        hits["review"] = [w for w in _REVIEW_DOC_HINT if w.lower() in ql][:3] + \
                         [w for w in _REVIEW_ACTION if w.lower() in ql][:2]
    kata = (hits["review"] or hits["multimodal"] or hits["rag"]
            or hits["nl2sql"] or hits["chitchat"])
    return _decide_skill(q), kata[:5]


# ══════════════════ 运行时路由（规则 + 实体证据）══════════════════════════════
def refine_with_entities(question: str, rule_skill: str, explicit: bool,
                         meta) -> tuple[str, str]:
    """用业务库实体证据修正规则判定，返回 (skill, reason)。

    - 规则判 nl2sql 但无实体证据 -> 降级 rag（泛词误判）
    - 规则兜底 rag 但实体强命中且带聚合意图 -> 升级 nl2sql（词表漏掉的问数）
    """
    cfg = get_config().orchestrator
    score, hits = score_biz_entities(question, meta)
    threshold = cfg.route_entity_min_score

    if rule_skill == "nl2sql" and score < threshold:
        return "rag", f"问数泛词命中但业务实体得分 {score} < {threshold}，降级知识问答"
    if (not explicit and rule_skill == "rag" and score >= threshold
            and _has_any(question, _AGG_INTENT)):
        return "nl2sql", f"业务实体命中 {hits[:3]}（得分 {score}）+ 聚合意图，升级问数"
    return rule_skill, f"实体得分 {score}"


async def decide_skill(question: str, *, use_meta: bool = True) -> tuple[str, dict]:
    """运行时路由：规则 -> 实体证据精修 -> 可选 LLM 兜底。返回 (skill, info)。"""
    rule_skill, explicit = _decide_skill_detail(question)
    info = {"rule": rule_skill, "explicit": explicit}

    cfg = get_config().orchestrator
    if not cfg.review_enabled and rule_skill == "review":
        rule_skill, explicit = "rag", False
        info["rule_downgraded"] = "review 未启用"

    skill = rule_skill
    if use_meta:
        meta = await aensure_route_meta()
        skill, reason = refine_with_entities(question, rule_skill, explicit, meta)
        info["reason"] = reason
        if skill != rule_skill:
            info["refined"] = f"{rule_skill}->{skill}"

    # 可选 LLM 精判：仅当规则落到默认 rag 且配置开启（实体证据也无力时）
    if skill == "rag" and not explicit and cfg.route_llm_fallback:
        refined = await _llm_refine_skill(question)
        if refined:
            skill = refined
            info["llm_refined"] = refined
    info["skill"] = skill
    return skill, info


async def route_node(state) -> dict:
    """AnswerGraph 的 route 节点：返回 skill/stage/trace 增量（条件边据此分发）。

    必须返回 dict（LangGraph 只合并返回值），仅原地改 state 不会传给条件边。
    """
    question = state.get("question", "")
    skill, info = await decide_skill(question)
    trace = list(state.get("trace", [])) + [
        f"route->{skill}" + (f"({info['refined']})" if info.get("refined") else "")
    ]
    out = {"skill": skill, "stage": f"route->{skill}", "trace": trace}
    if info.get("reason"):
        out["route_info"] = info
    return out


async def _llm_refine_skill(question: str) -> str | None:
    """规则与实体证据均无结论时用 LLM 做分类精判；失败返回 None（回退原规则）。"""
    try:
        from services.llm_ext import chat_json

        cfg = get_config()
        model = cfg.orchestrator.route_model or cfg.llm.model
        payload, _ = await chat_json(
            [
                {"role": "system", "content":
                    "判断用户的问题是希望查询结构化经营数据(nl2sql)、检索文档知识库(rag)、"
                    "读取图片/扫描件(multimodal)、审查合同文档(review)还是闲聊(chitchat)。"
                    "只输出 JSON: {\"skill\": \"rag\"}"},
                {"role": "user", "content": question},
            ],
            model=model, temperature=0.0, stage="orchestrator.route",
        )
        skill = str(payload.get("skill", "")).strip().lower()
        return skill if skill in SKILLS else None
    except Exception:  # noqa: BLE001
        return None
