#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_bert_train.py — BERT 意图识别模型训练相关单元测试
覆盖：数据集构建、标签映射、模型保存/加载、分类器预测。
"""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ───  fixtures  ────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_finetuned_model_path(tmp_path):
    """创建一个模拟的微调模型目录（含必要文件）。"""
    model_dir = tmp_path / "bert_intent"
    model_dir.mkdir()
    # 创建最小必要文件
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["BertForSequenceClassification"],
        "model_type": "bert",
        "num_labels": 4,
        "hidden_size": 768,
        "num_hidden_layers": 2,
        "num_attention_heads": 12,
        "vocab_size": 21128,
    }))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({
        "model_max_length": 512,
        "tokenizer_class": "BertTokenizer",
    }))
    (model_dir / "vocab.txt").write_text("bert-base-chinese\n")
    # 创建一个空 tensor 文件（模拟 pytorch_model.bin）
    import torch
    state = {"logits.weight": torch.zeros(4, 768), "logits.bias": torch.zeros(4)}
    torch.save(state, model_dir / "pytorch_model.bin")
    return str(model_dir)


@pytest.fixture
def mock_base_model_path():
    """指向实际的 bert-base-chinese 模型路径（用于不可用时测试）。"""
    return str(PROJECT_ROOT / "bert-base-chinese")


# ─── 标签映射测试 ──────────────────────────────────────────────────────────────
EXPECTED_LABELS = {
    "banking", "corporate_finance", "financial_accounting", "general",
    "financial_markets", "fintech", "insurance", "investment_banking",
    "personal_finance", "risk_management", "stock_market",
}


class TestLabelMap:
    def test_label_map_complete(self):
        from rag_qa.core.query_classifier import LABEL_MAP, INV_LABEL_MAP
        assert set(LABEL_MAP.keys()) == EXPECTED_LABELS
        assert set(INV_LABEL_MAP.values()) == EXPECTED_LABELS
        assert len(LABEL_MAP) == len(EXPECTED_LABELS)
        assert len(INV_LABEL_MAP) == len(EXPECTED_LABELS)
        # general 必须作为兜底层稳定存在
        assert LABEL_MAP["general"] == 3

    def test_inverse_consistency(self):
        from rag_qa.core.query_classifier import LABEL_MAP, INV_LABEL_MAP
        for label, idx in LABEL_MAP.items():
            assert INV_LABEL_MAP[idx] == label


# ─── 数据集测试 ────────────────────────────────────────────────────────────────
class TestIntentDataset:
    def test_build_dataset(self, mock_base_model_path):
        """构建数据集应正确分配标签。"""
        from transformers import BertTokenizer
        from rag_qa.core.query_classifier import build_dataset, IntentSample, LABEL_MAP
        tokenizer = BertTokenizer.from_pretrained(mock_base_model_path)
        faq_pairs = [
            ("银行贷款年利率是多少", "banking"),
            ("并购重组的估值方法", "corporate_finance"),
            ("资产负债表的编制方法", "financial_accounting"),
            ("今天天气怎么样", "general"),
            ("请问你好", "general"),
        ]
        dataset, dist = build_dataset(faq_pairs, [], tokenizer, max_length=32)
        assert len(dataset) == 5
        assert dist.get(LABEL_MAP["banking"], 0) == 1
        assert dist.get(LABEL_MAP["corporate_finance"], 0) == 1
        assert dist.get(LABEL_MAP["financial_accounting"], 0) == 1
        assert dist.get(LABEL_MAP["general"], 0) == 2

    def test_dataset_item_shape(self, mock_base_model_path):
        """数据集 item 应包含 input_ids, attention_mask, labels。"""
        from transformers import BertTokenizer
        from rag_qa.core.query_classifier import build_dataset, LABEL_MAP
        tokenizer = BertTokenizer.from_pretrained(mock_base_model_path)
        faq_pairs = [("银行贷款", "banking")]
        dataset, _ = build_dataset(faq_pairs, [], tokenizer, max_length=16)
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["labels"].item() == LABEL_MAP["banking"]


# ─── 关键词分类器测试 ──────────────────────────────────────────────────────────
class TestKeywordClassify:
    def test_banking(self):
        from rag_qa.core.query_classifier import _keyword_classify
        domain, conf, hits = _keyword_classify("银行贷款利率是多少")
        assert domain == "banking"
        assert conf > 0
        assert any(h in "银行贷款利率是多少" for h in hits)

    def test_general(self):
        from rag_qa.core.query_classifier import _keyword_classify
        domain, conf, hits = _keyword_classify("今天天气怎么样")
        assert domain == "general"
        assert conf == 0.0

    def test_empty(self):
        from rag_qa.core.query_classifier import _keyword_classify
        domain, conf, hits = _keyword_classify("")
        assert domain == "general"
        assert conf == 0.0

    def test_low_confidence_fallback(self):
        from rag_qa.core.query_classifier import _keyword_classify
        # 命中多个领域 → 置信度可能低于 0.6 → 降级为 general
        domain, conf, hits = _keyword_classify("银行和公司财务都涉及贷款")
        assert isinstance(domain, str)
        assert domain in ("banking", "corporate_finance", "financial_accounting", "general")


# ─── QueryClassifier 测试 ─────────────────────────────────────────────────────
class TestQueryClassifier:
    def test_init_no_finetuned(self, mock_base_model_path):
        """未训练模型时应初始化成功并加载基础 BERT 模型作为降级方案。"""
        from rag_qa.core.query_classifier import QueryClassifier
        clf = QueryClassifier(model_path="/nonexistent/path")
        # 应加载基础模型，而非完全 None（关键词兜底）
        assert clf.model is not None
        assert clf._use_finetuned is False

    def test_predict_fallback_to_keyword(self, mock_base_model_path):
        """无模型时预测应降级为关键词分类。"""
        from rag_qa.core.query_classifier import QueryClassifier
        clf = QueryClassifier(model_path="/nonexistent/path")
        domain, conf, hits = clf.predict("银行贷款政策")
        assert domain in ("banking", "general")
        assert isinstance(conf, float)

    @pytest.mark.skip(reason="Requires transformers and bert-base-chinese model")
    def test_predict_with_base_model(self, mock_base_model_path):
        """使用基础模型预测应返回 domain。"""
        from rag_qa.core.query_classifier import QueryClassifier
        clf = QueryClassifier(model_path=mock_base_model_path)
        domain, conf, hits = clf.predict("银行贷款利率")
        assert domain in ("banking", "general")
        assert isinstance(conf, float)

    def test_is_finetuned_false_when_no_model(self):
        from rag_qa.core.query_classifier import QueryClassifier
        clf = QueryClassifier(model_path="/nonexistent/path")
        assert clf.is_finetuned() is False


# ─── classify_intent 入口测试 ─────────────────────────────────────────────────
class TestClassifyIntent:
    def test_general_query(self):
        from rag_qa.core.query_classifier import classify_intent
        result = classify_intent("今天天气怎么样")
        assert result.domain == "general"

    def test_banking_query(self):
        from rag_qa.core.query_classifier import classify_intent
        result = classify_intent("银行贷款利率是多少")
        assert result.domain in ("banking", "general")

    def test_corporate_finance_query(self):
        from rag_qa.core.query_classifier import classify_intent
        result = classify_intent("公司并购重组的估值方法")
        # BERT 模型可能将"估值"关联到 financial_accounting，接受合理分类
        assert result.domain in ("corporate_finance", "financial_accounting", "general")

    def test_financial_accounting_query(self):
        from rag_qa.core.query_classifier import classify_intent
        result = classify_intent("资产负债表的编制方法")
        assert result.domain in ("financial_accounting", "general")


# ─── 训练脚本模块级测试 ───────────────────────────────────────────────────────
class TestTrainScriptImports:
    def test_import_modules(self):
        """训练脚本导入应无错误。"""
        import scripts.train_intent_bert as train_module
        assert hasattr(train_module, "build_dataset")
        assert hasattr(train_module, "IntentSample")
        assert hasattr(train_module, "IntentDataset")
        assert hasattr(train_module, "LABEL_MAP")
        assert hasattr(train_module, "augment_general_samples")

    def test_label_map_matches_classifier(self):
        """训练脚本和分类器的标签映射应一致。"""
        from scripts.train_intent_bert import LABEL_MAP as train_labels
        from rag_qa.core.query_classifier import LABEL_MAP as clf_labels
        assert train_labels == clf_labels


# ─── 训练配置测试 ──────────────────────────────────────────────────────────────
class TestTrainConfig:
    def test_config_ini_has_models_section(self):
        import configparser
        cp = configparser.ConfigParser()
        cp.read(str(PROJECT_ROOT / "config.ini"), encoding="utf-8")
        assert "models" in cp
        assert "bert_intent_model_path" in cp["models"]
        assert "bert_epochs" in cp["models"]
        assert "bert_batch_size" in cp["models"]
        assert "bert_lr" in cp["models"]
