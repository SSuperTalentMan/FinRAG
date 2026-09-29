#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/repair_missing_faq.py — 修复 MySQL↔Milvus 对账发现的缺失 FAQ 向量

背景：对账（services/reconciliation.py）发现部分 MySQL finance_faq 问题在
Milvus 中无对应向量（多为原 3 领域 banking/corporate_finance/financial_accounting，
因 load_new_data.py 为保护存量向量刻意不写原领域）。

本脚本仅做「补齐」：为 MySQL 有、Milvus 无向量的问题生成 BGE-M3 向量并写入，
不删除、不覆盖任何既有数据，可重复执行（幂等）。

用法：
    python scripts/repair_missing_faq.py              # 补齐全部缺失
    python scripts/repair_missing_faq.py --dry-run    # 仅预览缺失，不写入
    python scripts/repair_missing_faq.py --batch 32   # 调整编码批次
"""
import os
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── CPU 性能优化（torch 导入前）────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["MKL_NUM_THREADS"]      = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import argparse

from config import get_config
from db.mysql import get_db
from db.milvus import get_milvus_client
from services.embedding import encode_batch_dense_sparse


def _mysql_faq_rows() -> dict[str, dict]:
    """MySQL finance_faq 全量行，按 question 去重索引。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT question, answer, category, type FROM finance_faq")
        rows = {}
        for r in cur.fetchall():
            q = str(r["question"]).strip()
            if q and q not in rows:  # 同名问题保留首条，避免重复向量
                rows[q] = {
                    "question": q,
                    "answer": r["answer"] or "",
                    "category": r["category"] or "",
                    "type": r["type"] or "",
                }
        return rows


def _milvus_faq_questions() -> set[str]:
    """Milvus 中现有 FAQ 问题集（无 kb_id 的行）。"""
    cfg = get_config()
    client = get_milvus_client()
    questions: set[str] = set()
    offset = 0
    while True:
        rows = client.query(
            collection_name=cfg.milvus.collection_name,
            filter="",
            limit=2000,
            offset=offset,
            output_fields=["kb_id", "question"],
        )
        if not rows:
            break
        for row in rows:
            if row.get("kb_id") is None:
                q = str(row.get("question") or "").strip()
                if q:
                    questions.add(q)
        offset += len(rows)
        if len(rows) < 2000:
            break
    return questions


def main():
    ap = argparse.ArgumentParser(description="补齐 MySQL 有而 Milvus 无向量的 FAQ")
    ap.add_argument("--dry-run", action="store_true", help="仅预览缺失数量，不写入")
    ap.add_argument("--batch", type=int, default=32, help="向量化批次大小")
    args = ap.parse_args()

    cfg = get_config()
    logger.info("Step 1: 读取 MySQL finance_faq ...")
    mysql_rows = _mysql_faq_rows()
    logger.info(f"  MySQL FAQ 共 {len(mysql_rows)} 条")

    logger.info("Step 2: 扫描 Milvus 现有 FAQ 向量 ...")
    milvus_qs = _milvus_faq_questions()
    logger.info(f"  Milvus FAQ 向量共 {len(milvus_qs)} 条")

    missing = sorted(set(mysql_rows) - milvus_qs)
    logger.info(f"缺失向量 {len(missing)} 条")
    if not missing:
        logger.info("无需修复，MySQL↔Milvus FAQ 已一致")
        return
    from collections import Counter
    dist = Counter(mysql_rows[q]["category"] for q in missing)
    for cat, n in dist.most_common():
        logger.info(f"  {cat}: {n}")

    if args.dry_run:
        logger.info("--dry-run，未写入任何数据")
        return

    logger.info("Step 3: BGE-M3 向量化并写入 Milvus ...")
    client = get_milvus_client()
    inserted = 0
    for i in range(0, len(missing), args.batch):
        batch_qs = missing[i:i + args.batch]
        texts = [f"{mysql_rows[q]['question']} {mysql_rows[q]['answer']}" for q in batch_qs]
        dense_list, sparse_list = encode_batch_dense_sparse(texts)
        data = [{
            "dense_vector":  dense_list[j],
            "sparse_vector": sparse_list[j],
            "text":          f"{mysql_rows[q]['question']}\n答：{mysql_rows[q]['answer']}",
            "category":      mysql_rows[q]["category"],
            "type":          mysql_rows[q]["type"],
            "question":      mysql_rows[q]["question"],
            "answer":        mysql_rows[q]["answer"],
        } for j, q in enumerate(batch_qs)]
        client.insert(collection_name=cfg.milvus.collection_name, data=data)
        inserted += len(data)
        logger.info(f"  已写入 {min(i + args.batch, len(missing))}/{len(missing)}")

    stats = client.get_collection_stats(cfg.milvus.collection_name)
    logger.info(f"修复完成：新增 {inserted} 条向量，集合总行数 {stats.get('row_count')}")

    # 重新对账确认
    logger.info("Step 4: 重新对账验证 ...")
    from services.reconciliation import run_reconciliation
    res = run_reconciliation()
    logger.info(f"对账结果: FAQ 缺失 {res.get('faq_missing', '?')} 条")


if __name__ == "__main__":
    main()
