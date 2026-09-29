#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_qa/core/query_classifier.py — BERT 意图分类器
支持训练/加载/预测。
- 若 rag_qa/models/bert_intent/ 存在，则加载微调模型（多分类，类别数见 LABEL_MAP）；
- 否则使用 bert-base-chinese 作为基础二分类器（通用知识 vs 专业咨询）；
- 最终将预测结果映射到领域分类（含新增的金融市场/金融科技/保险/投行/个人理财/风险/股票等领域）。
"""

import os
import threading
from dataclasses import dataclass
from loguru import logger

# 优先使用 BERT；若无 transformers 则静默跳过
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    _HAS_BERT = True
except ImportError:
    _HAS_BERT = False


@dataclass
class IntentResult:
    domain: str          # banking / corporate_finance / financial_accounting / financial_markets /
                         # fintech / insurance / investment_banking / personal_finance /
                         # risk_management / stock_market / general
    confidence: float    # 0.0 ~ 1.0
    keywords_hit: list[str]


# ─── 关键词兜底分类器 ───────────────────────────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "banking": [
        "银行", "信贷", "贷款", "存款", "利率", "汇兑", "结算", "票据",
        "信用卡", "借记卡", "透支", "同业", "拆借",
        "央行", "CBDC", "支付", "清算", "ATM", "网银",
        "普惠金融", "小微贷款", "供应链金融", "贸易融资", "保理",
        "资本充足率", "巴塞尔", "存贷比", "不良资产", "坏账", "MRA",
        "PSB", "商业银行", "政策性银行", "农信社", "村镇银行",
    ],
    "corporate_finance": [
        "公司金融", "企业融资", "融资", "股权", "债权", "并购", "重组",
        "IPO", "上市", "退市", "估值", "尽职调查",
        "现金流", "资本结构", "加权平均资本成本", "WACC", "NPV", "IRR",
        "DCF", "折现", "现值", "资本预算", "投资回收期",
        "股东权益", "董事会", "治理", "代理问题", "激励",
        "股利", "回购", "股息",
        "跨国公司", "跨国经营", "海外投资", "FDI",
    ],
    "financial_accounting": [
        "会计", "财务", "资产负债表", "利润表", "现金流量表", "所有者权益",
        "折旧", "摊销", "坏账准备", "存货", "应收账款", "应付账款",
        "审计", "会计准则", "GAAP", "IFRS", "审计意见",
        "财务报告", "合并报表", "抵销", "商誉",
        "收入确认", "成本核算", "费用", "税金",
        "公允价值", "历史成本", "权责发生制", "收付实现制",
    ],
    "financial_markets": [
        "金融市场", "资本市场", "证券市场", "交易所", "债券",
        "货币政策", "汇率", "衍生品", "期货", "期权", "对冲基金", "共同基金",
        "机构投资者", "市场微观结构", "有效市场", "波动率", "做市商",
        "一级市场", "二级市场", "收益率曲线", "信用评级", "资产定价",
    ],
    "fintech": [
        "金融科技", "数字支付", "移动支付", "第三方支付", "区块链", "加密货币",
        "比特币", "分布式账本", "智能合约", "开放银行", "监管科技", "保险科技",
        "网贷", "P2P", "众筹", "数字钱包", "稳定币", "央行数字货币", "跨境支付",
        "开放金融", "API银行", "嵌入式金融",
    ],
    "insurance": [
        "保险", "承保", "理赔", "保费", "保单", "精算", "寿险", "财险",
        "再保险", "年金", "免责条款", "保险责任", "健康险", "车险", "意外险",
        "偿付能力", "道德风险", "逆向选择", "退保", "核保",
    ],
    "investment_banking": [
        "投行", "投资银行", "承销", "资产重组", "私募股权", "风险投资",
        "路演", "招股说明书", "绿鞋", "杠杆收购", "过桥贷款", "银团贷款",
        "结构化融资", "资产证券化", "买方", "卖方", "做市", "财务顾问",
    ],
    "personal_finance": [
        "个人理财", "财富管理", "资产配置", "投资组合", "基金定投", "退休规划",
        "养老", "储蓄", "信用卡管理", "债务管理", "税务筹划", "保险规划",
        "财务自由", "复利", "分散投资", "理财目标", "现金流规划", "教育金", "遗产规划",
    ],
    "risk_management": [
        "风险管理", "风险识别", "风险评估", "风险敞口", "风险对冲", "在险价值",
        "VaR", "信用风险", "市场风险", "操作风险", "流动性风险", "压力测试",
        "风险偏好", "风险限额", "套期保值", "风险矩阵", "情景分析", "尾部风险", "巴塞尔协议",
    ],
    "stock_market": [
        "股票", "股市", "A股", "港股", "美股", "上市公司", "股价", "市值",
        "涨停", "跌停", "成交量", "换手率", "市盈率", "市净率", "分红",
        "除权", "除息", "股息率", "蓝筹股", "成长股", "指数", "大盘", "板块",
        "北向资金", "量化交易", "游资", "散户",
    ],
}

# 高优先级领域覆盖规则：处理"跨词歧义"固定搭配
# 这类词组在字面上同时命中多个领域关键词，BERT 微调模型易被表层词主导而误路由，
# 例如「存款保险」是银行监管制度（Deposit Insurance），不属于保险业经营范畴。
# 规则命中时直接覆盖 BERT 结果，避免严格领域过滤把正确内容挡在检索之外。
DOMAIN_OVERRIDE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("存款保险", "存款保险条例", "存款保险基金", "存款偿付", "存款保险费率"), "banking"),
    (("征信", "信用报告", "征信异议", "个人信用信息"), "banking"),
    # 数字人民币/金融科技文档落在 fintech 域，BERT 对「重大调整」等措辞会判成 banking
    (("数字人民币", "数字存款货币", "e-CNY", "央行数字货币"), "fintech"),
    (("金融科技", "金融数字化转型"), "fintech"),
    # 投资教育知识库(kb=4, domain=investment_banking)内容路由：
    # 「注册制改革」系列投教问答、投资者教育基地文章均落在此域。BERT 微调模型易把
    # 「注册制」判为 financial_markets、「投资者教育」判为 stock_market，导致严格领域
    # 过滤把 doc_4（注册制投教问答）/ doc_31（投资者教育文章）挡在检索之外。固定搭配
    # 覆盖修正路由，确保这类查询能命中 kb=4。
    (("股票发行注册制", "注册制改革", "全面实行股票发行注册制", "发行注册制", "注册制下", "注册制试点"), "investment_banking"),
    (("投资者教育", "投教", "投资者教育基地"), "investment_banking"),
    # 财务报表类术语在语义上无歧义地属于财务会计领域；BERT 微调模型对
    # 「资产负债表的编制方法」等表述易误判为 risk_management/stock_market，
    # 固定搭配覆盖修正路由，避免严格领域过滤把正确内容挡在检索之外。
    (("资产负债表", "利润表", "现金流量表", "所有者权益", "财务报表"), "financial_accounting"),
]


def _override_domain(query: str) -> tuple[str | None, list[str]]:
    """按固定搭配规则判定领域，命中则返回 (domain, 命中词)，否则 (None, [])。"""
    for phrases, domain in DOMAIN_OVERRIDE_RULES:
        matched = [p for p in phrases if p in query]
        if matched:
            return domain, matched
    return None, []


# 领域标签映射（general 固定在索引 3，作为未知/通用兜底层；新增领域追加其后）
LABEL_MAP: dict[str, int] = {
    "banking":             0,
    "corporate_finance":   1,
    "financial_accounting": 2,
    "general":             3,
    "financial_markets":   4,
    "fintech":             5,
    "insurance":           6,
    "investment_banking":   7,
    "personal_finance":    8,
    "risk_management":     9,
    "stock_market":       10,
}
GENERAL_LABEL: int = LABEL_MAP["general"]

INV_LABEL_MAP: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


def _keyword_classify(query: str) -> tuple[str, float, list[str]]:
    """纯关键词规则分类（兜底）。"""
    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in query]
        scores[domain] = len(matched)
        hits[domain] = matched

    if not any(scores.values()):
        return "general", 0.0, []

    max_domain = max(scores, key=scores.get)
    max_score  = scores[max_domain]
    total_hits = sum(scores.values())
    confidence = round(max_score / total_hits, 4) if total_hits > 0 else 0.0

    if confidence < 0.6:
        return "general", confidence, hits[max_domain]
    return max_domain, confidence, hits[max_domain]


# ─── BERT 分类器 ───────────────────────────────────────────────────────────────
class QueryClassifier:
    """
    BERT 意图分类器。
    - 若 rag_qa/models/bert_intent 存在，则加载微调模型（四分类）；
    - 否则使用 bert-base-chinese 做基础二分类，再结合关键词映射到四领域；
    - 训练模型不可用时自动降级为关键词分类。
    """

    def __init__(self, model_path: str = ""):
        from config import get_config
        cfg = get_config()
        self.model_path = model_path or cfg.models.bert_intent_model_path
        self.tokenizer = None
        self.model     = None
        self.device    = self._device()
        self._use_finetuned = self._has_finetuned_model()

        if _HAS_BERT:
            self.load_model()
        else:
            logger.warning("transformers 未安装，QueryClassifier 将使用关键词兜底")

    def _has_finetuned_model(self) -> bool:
        """检查是否已保存微调模型（支持 safetensors 和 pytorch_model.bin）。"""
        return os.path.exists(os.path.join(self.model_path, "model.safetensors")) or \
               os.path.exists(os.path.join(self.model_path, "pytorch_model.bin"))

    @staticmethod
    def _device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> None:
        if not _HAS_BERT:
            return
        try:
            if self._use_finetuned:
                logger.info(f"加载微调 BERT 模型: {self.model_path}")
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_path, num_labels=len(LABEL_MAP)
                )
            else:
                # 无微调模型，用基础 bert-base-chinese 做二分类
                base_path = self._get_base_model_path()
                logger.info(f"加载基础 BERT 模型: {base_path}（二分类，关键词辅助映射）")
                self.tokenizer = BertTokenizer.from_pretrained(base_path)
                self.model = BertForSequenceClassification.from_pretrained(
                    base_path, num_labels=2
                )
                self._base_model = True
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"BERT 意图分类器加载成功: device={self.device}")
        except Exception as e:
            logger.error(f"BERT 模型加载失败，使用关键词兜底: {e}")
            self.model = None

    def _get_base_model_path(self) -> str:
        """获取基础 BERT 模型路径。"""
        from config import get_config
        cfg = get_config()
        base = cfg.models.bert_base_model_path
        if os.path.exists(base):
            return base
        return "bert-base-chinese"

    def predict(self, query: str) -> tuple[str, float, list[str]]:
        """
        对单条查询做意图分类。
        返回 (domain, confidence, keywords_hit)。
        - 微调模型：直接用 BERT 预测结果
        - 基础模型：二分类结果 + 关键词映射
        - 均失败时：关键词兜底
        """
        # 1) 固定搭配覆盖规则（优先于 BERT，修正跨词歧义误路由）
        ov_domain, ov_hits = _override_domain(query)
        if ov_domain is not None:
            logger.debug(f"领域覆盖规则命中: {ov_domain} <- {ov_hits}")
            return ov_domain, 0.99, ov_hits

        if self.model is not None and self.tokenizer is not None:
            bert_domain, bert_conf = self._bert_predict(query)
            if bert_domain is not None:
                # 微调模型直接返回；但在置信度很低时用关键词结果兜底，
                # 避免政策类 Query 被误路由到错误领域（影响严格领域过滤下的检索）。
                if self._use_finetuned and bert_conf < 0.5:
                    kw_domain, kw_conf, kw_hits = _keyword_classify(query)
                    if kw_domain != "general":
                        return kw_domain, max(bert_conf, kw_conf), kw_hits
                return bert_domain, bert_conf, []

        # BERT 不可用时，降级为关键词分类
        domain, confidence, hits = _keyword_classify(query)
        logger.debug(f"BERT 不可用，降级关键词分类: domain={domain}")
        return domain, confidence, hits

    def _bert_predict(self, query: str) -> tuple[str | None, float]:
        """
        BERT 预测，返回 (domain, confidence) 或 (None, 0.0) 若失败。
        - 微调模型：直接四分类输出
        - 基础模型：二分类（通用知识/专业咨询）+ 关键词辅助映射
        """
        if not _HAS_BERT or self.model is None or self.tokenizer is None:
            return None, 0.0

        try:
            encoding = self.tokenizer(
                query, truncation=True, padding=True,
                max_length=64, return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = float(torch.max(probs).item())

            if self._use_finetuned:
                # 四分类：直接映射
                domain = INV_LABEL_MAP.get(pred_idx, "general")
                return domain, confidence
            else:
                # 二分类（通用知识=0 / 专业咨询=1）+ 关键词辅助
                if pred_idx == 1:
                    # 专业咨询 → 关键词确定具体领域
                    domain, conf, _ = _keyword_classify(query)
                    return domain, confidence
                else:
                    # 通用知识 → 直接 general
                    return "general", confidence

        except Exception as e:
            logger.warning(f"BERT 预测失败: {e}")
            return None, 0.0

    def is_finetuned(self) -> bool:
        """返回是否加载了微调模型。"""
        return self._use_finetuned


# ─── 数据集类 ──────────────────────────────────────────────────────────────────
class IntentSample:
    """单条意图训练样本。"""
    __slots__ = ("text", "label")

    def __init__(self, text: str, label: int):
        self.text = text
        self.label = label


class IntentDataset:
    """PyTorch 意图识别数据集。"""

    def __init__(self, samples: list, tokenizer, max_length: int = 64):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(sample.label, dtype=torch.long),
        }


def build_dataset(
    faq_pairs: list,
    augmented_general: list,
    tokenizer,
    max_length: int = 64,
):
    """
    从 FAQ 对和补充样本构建训练数据集。
    返回 (IntentDataset, label_distribution_dict)。
    """
    samples: list = []
    for question, category in faq_pairs:
        label = LABEL_MAP.get(category, GENERAL_LABEL)
        samples.append(IntentSample(text=question, label=label))
    for question in augmented_general:
        samples.append(IntentSample(text=question, label=LABEL_MAP["general"]))

    import random
    random.seed(42)
    random.shuffle(samples)

    dist: dict = {}
    for s in samples:
        dist[s.label] = dist.get(s.label, 0) + 1
    return IntentDataset(samples, tokenizer, max_length), dist


# ─── 全局单例 ──────────────────────────────────────────────────────────────────
_classifier: QueryClassifier | None = None
_classifier_lock = threading.Lock()  # 双检锁：避免并发首请求重复加载 BERT 模型


def get_classifier() -> QueryClassifier:
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = QueryClassifier()
    return _classifier


def is_classifier_ready() -> bool:
    """BERT 意图分类器是否就绪（只读探测，不触发加载）。

    未就绪时系统会自动降级为关键词分类，故健康检查将其视为可降级项。
    """
    c = _classifier
    return c is not None and c.model is not None


def classify_intent(query: str) -> IntentResult:
    """
    意图分类入口，返回 IntentResult(domain, confidence, keywords_hit)。
    兼容原有 services.intent.classify_intent 的签名。
    """
    classifier = get_classifier()
    domain, confidence, hits = classifier.predict(query)
    logger.debug(f"意图识别: domain={domain}, confidence={confidence:.4f}, hits={hits}")
    return IntentResult(domain=domain, confidence=confidence, keywords_hit=hits)
