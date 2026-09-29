#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_intent_bert.py — BERT 意图识别模型训练脚本

功能：
  1. 从 MySQL finance_faq 表读取 FAQ 数据作为训练语料
  2. 对无领域标签的问题补充生成 general 标签（关键词兜底）
  3. 使用 bert-base-chinese 微调四分类意图识别模型
  4. 保存模型至 rag_qa/models/bert_intent/

用法：
    python scripts/train_intent_bert.py
    python scripts/train_intent_bert.py --epochs 5 --batch-size 16
    python scripts/train_intent_bert.py --model-path ./bert-base-chinese
"""

import os
import sys
import argparse
import json
from pathlib import Path
from dataclasses import dataclass

# ─── CPU 性能优化 ───────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["MKL_NUM_THREADS"]      = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
import pymysql
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from loguru import logger

# ─── 路径常量 ───────────────────────────────────────────────────────────────────
# 始终基于本文件位置推导（脚本位于 scripts/ 下，parent.parent 即项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config

# ─── 日志配置 ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)
logger.add(
    str(LOG_DIR / "bert_train.log"),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
)

# ─── 标签映射 ───────────────────────────────────────────────────────────────────
# general 固定在索引 3（作为未知/通用兜底层），新增金融领域追加其后，与
# rag_qa/core/query_classifier.py 中的 LABEL_MAP 保持一致。
LABEL_MAP = {
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
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
GENERAL_LABEL = LABEL_MAP["general"]

# ─── 通用问题关键词（用于补充 general 样本）──────────────────────────────────────
GENERAL_KEYWORDS = [
    "什么", "怎么", "如何", "为什么", "哪里", "多少", "几", "哪个",
    "好吗", "谢谢", "你好", "请问", "可以", "需要", "应该", "可以",
    "可以告诉我", "能帮我", "请问一下", "怎样", "何时",
]

# ─── 补充的通用问题模板 ─────────────────────────────────────────────────────────
GENERAL_SAMPLES = [
    "请问今天天气怎么样",
    "你好，我想咨询一下",
    "谢谢你的帮助",
    "这是什么意思",
    "为什么会有这种情况",
    "怎么才能解决这个问题",
    "大概需要多长时间",
    "在哪里可以办理",
    "一共有多少种方式",
    "请问还有别的办法吗",
    "您好，能帮我查一下吗",
    "怎么样才能更快地办理",
    "为什么我的账户被冻结了",
    "什么时候可以到账",
    "能不能取消这个操作",
    "请问有什么优惠政策",
    "大概需要多少钱",
    "在哪里可以查到相关信息",
    "怎么查看我的余额",
    "请问我的账户状态是否正常",
]


@dataclass
class IntentSample:
    text: str
    label: int


class IntentDataset(Dataset):
    """意图识别数据集。"""

    def __init__(self, samples: list[IntentSample], tokenizer: BertTokenizer, max_length: int = 64):
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


def load_faq_data() -> list[tuple[str, str]]:
    """从 MySQL 读取所有 FAQ 问答对，返回 [(question, category), ...]。"""
    cfg = get_config()
    logger.info(f"连接 MySQL {cfg.mysql.host}:{cfg.mysql.port}/{cfg.mysql.database}")
    conn = pymysql.connect(
        host=cfg.mysql.host,
        port=cfg.mysql.port,
        user=cfg.mysql.user,
        password=cfg.mysql.password,
        database=cfg.mysql.database,
        charset=cfg.mysql.charset,
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    cur.execute("SELECT question, category FROM finance_faq")
    rows = cur.fetchall()
    conn.close()
    logger.info(f"从 MySQL 读取 {len(rows)} 条 FAQ 数据")
    return [(r["question"], r["category"]) for r in rows]


def augment_general_samples(min_general: int = 30) -> list[str]:
    """
    当 general 类别样本不足时，用模板生成补充样本。
    返回补充的 question 列表。
    """
    existing = load_faq_data()
    general_count = sum(1 for _, cat in existing if cat == "general")
    logger.info(f"现有 general 样本数: {general_count}")

    # 取每个领域的样本数，用于均衡计算
    domain_counts = {}
    for _, cat in existing:
        domain_counts[cat] = domain_counts.get(cat, 0) + 1
    # general 样本数不超过其他领域平均值的 80%，避免过拟合 general
    other_avg = sum(v for k, v in domain_counts.items() if k != "general") / max(1, len([v for k, v in domain_counts.items() if k != "general"]))
    max_general = int(other_avg * 0.8)
    min_general = min(min_general, max_general)

    if general_count >= min_general:
        return []

    need = min_general - general_count
    augmented = []
    for i in range(need):
        template = GENERAL_SAMPLES[i % len(GENERAL_SAMPLES)]
        augmented.append(template)
    logger.info(f"补充 general 样本 {len(augmented)} 条（均衡后最多 {min_general} 条）")
    return augmented


def load_macro_samples(path: str | None = None) -> list[tuple[str, int]]:
    """加载宏观政策类补充样本（data/intent_macro_samples.jsonl）。返回 [(text, label_id)]。"""
    if not path:
        path = str(PROJECT_ROOT / "data" / "intent_macro_samples.jsonl")
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        logger.warning(f"宏观政策样本文件不存在: {p}")
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("text") and isinstance(o.get("label"), int):
            out.append((o["text"], o["label"]))
    logger.info(f"加载宏观政策补充样本 {len(out)} 条（{p}）")
    return out


def build_dataset(
    faq_pairs: list[tuple[str, str]],
    augmented_general: list[str],
    tokenizer: BertTokenizer,
    max_length: int = 64,
    macro_samples: list[tuple[str, int]] | None = None,
) -> tuple[IntentDataset, dict[int, int]]:
    """构建训练数据集，返回 (dataset, label_distribution)。"""
    samples: list[IntentSample] = []

    for question, category in faq_pairs:
        label = LABEL_MAP.get(category, GENERAL_LABEL)  # 未知类别归入 general
        samples.append(IntentSample(text=question, label=label))

    # 补充 general 样本
    for question in augmented_general:
        samples.append(IntentSample(text=question, label=LABEL_MAP["general"]))

    # 补充宏观政策类样本（提升政策类 Query 的意图路由准确率，避免误分 investment_banking）
    if macro_samples:
        for text, label in macro_samples:
            samples.append(IntentSample(text=text, label=label))

    # 打乱顺序
    import random
    random.seed(42)
    random.shuffle(samples)

    # 统计分布
    dist: dict[int, int] = {}
    for s in samples:
        dist[s.label] = dist.get(s.label, 0) + 1
    logger.info(f"数据集总样本数: {len(samples)}")
    for label_id, count in sorted(dist.items()):
        logger.info(f"  {INV_LABEL_MAP.get(label_id, 'unknown')}: {count} 条")

    dataset = IntentDataset(samples, tokenizer, max_length)
    return dataset, dist


def train_model(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    warmup_steps: int,
    output_dir: str,
) -> BertForSequenceClassification:
    """训练模型，返回训练好的模型。"""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.to(device)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 50 == 0:
                avg = total_loss / (step + 1)
                logger.info(f"  Epoch {epoch}/{epochs} Step {step+1}/{len(train_loader)}  loss={avg:.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{epochs}  avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(output_dir)
            logger.info(f"  保存最佳模型 (loss={best_loss:.4f}) → {output_dir}")

    return model


def evaluate_model(
    model: BertForSequenceClassification,
    dataset: IntentDataset,
    device: str,
    batch_size: int = 32,
) -> dict:
    """评估模型，返回 classification_report 风格的指标。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())

    # 计算 per-label 准确率
    from collections import defaultdict
    label_tp = defaultdict(int)
    label_total = defaultdict(int)
    for pred, true in zip(all_preds, all_labels):
        label_total[true] += 1
        if pred == true:
            label_tp[true] += 1

    overall_correct = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
    overall_acc = overall_correct / len(all_labels) if all_labels else 0.0

    report = {
        "overall_accuracy": round(overall_acc, 4),
        "total_samples": len(all_labels),
        "per_label": {},
    }
    for label_id, total in sorted(label_total.items()):
        tp = label_tp[label_id]
        report["per_label"][INV_LABEL_MAP.get(label_id, f"label_{label_id}")] = {
            "accuracy": round(tp / total, 4) if total > 0 else 0.0,
            "count": total,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="BERT 意图识别模型训练")
    parser.add_argument("--epochs",      type=int,   default=5,   help="训练轮数")
    parser.add_argument("--batch-size",  type=int,   default=16,  help="批次大小")
    parser.add_argument("--lr",          type=float, default=2e-5, help="学习率")
    parser.add_argument("--max-length",  type=int,   default=64,  help="最大序列长度")
    parser.add_argument("--model-path",  type=str,   default="",  help="BERT 基础模型路径（空则用 bert-base-chinese）")
    parser.add_argument("--output-dir",  type=str,   default="",  help="模型输出目录（空则用 rag_qa/models/bert_intent）")
    parser.add_argument("--min-general", type=int,   default=50,  help="general 类别最小样本数")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BERT 意图识别模型训练启动")
    logger.info("=" * 60)
    logger.info(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"  max_length={args.max_length}, min_general={args.min_general}")

    # 确定路径（优先命令行参数，其次配置文件）
    cfg = get_config()
    base_model_path  = args.model_path or cfg.models.bert_base_model_path
    output_dir       = args.output_dir or str(PROJECT_ROOT / "rag_qa" / "models" / "bert_intent")
    os.makedirs(output_dir, exist_ok=True)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"训练设备: {device}")

    # 加载分词器
    logger.info(f"加载分词器: {base_model_path}")
    tokenizer = BertTokenizer.from_pretrained(base_model_path)
    logger.info("分词器加载完成")

    # 加载数据
    faq_pairs = load_faq_data()
    augmented = augment_general_samples(args.min_general)
    macro = load_macro_samples()
    dataset, dist = build_dataset(faq_pairs, augmented, tokenizer, args.max_length, macro_samples=macro)

    # 构建 DataLoader
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型
    num_labels = len(LABEL_MAP)
    logger.info(f"初始化模型: {base_model_path} (num_labels={num_labels})")
    model = BertForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=num_labels,
    )

    # 训练
    logger.info("开始训练...")
    warmup_steps = max(10, len(train_loader) * args.epochs // 10)
    model = train_model(model, train_loader, device, args.epochs, args.lr, warmup_steps, output_dir)

    # 评估
    logger.info("开始评估...")
    report = evaluate_model(model, dataset, device)
    logger.info(f"总体准确率: {report['overall_accuracy']:.4f}")
    for label_name, metrics in report["per_label"].items():
        logger.info(f"  {label_name}: accuracy={metrics['accuracy']:.4f}, count={metrics['count']}")

    # 保存 tokenizer（已包含在 save_pretrained 中）
    tokenizer.save_pretrained(output_dir)

    # 保存训练配置
    config_path = Path(output_dir) / "train_config.json"
    config = {
        "base_model":   base_model_path,
        "output_dir":   output_dir,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "max_length":   args.max_length,
        "num_labels":   num_labels,
        "label_map":    LABEL_MAP,
        "train_samples": report["total_samples"],
        "overall_accuracy": report["overall_accuracy"],
        "per_label":    report["per_label"],
        "created_at":   torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"训练配置已保存: {config_path}")

    logger.info("=" * 60)
    logger.info("训练完成！模型已保存至: " + output_dir)
    logger.info("=" * 60)
    return report


if __name__ == "__main__":
    main()
