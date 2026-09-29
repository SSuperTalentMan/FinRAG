#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/load_new_data.py — 将清洗翻译后的新领域数据入库

  (1) MySQL finance_faq：按领域写入「高频问答」样本（每个领域涉及一点），
      不 DROP 整表，仅对涉及的领域做幂等覆盖（先删该领域旧行再插），不影响原有 3 个领域。
      该表同时驱动 BERT 意图训练 与 BM25 检索索引。
  (2) Milvus finrag_faq：增量写入全部翻译后的新领域向量（稠密+稀疏，BGE-M3），
      不删除集合、不触碰已有 3 个领域数据；集合不存在时按既定 schema 自动创建。

用法：
    python scripts/load_new_data.py                 # 默认：每领域 MySQL 40 条，Milvus 全量
    python scripts/load_new_data.py --mysql-per 60   # 提高 MySQL 高频样本量
    python scripts/load_new_data.py --domain financial_markets,stock_market
    python scripts/load_new_data.py --milvus-batch 64

依赖：config.py、pymysql、pymilvus、FlagEmbedding、scipy、loguru
"""
import os
import sys
import json
import glob
import random
import argparse
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── CPU 性能优化（torch 导入前）────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["MKL_NUM_THREADS"]      = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

from config import get_config
_cfg = get_config()

import pymysql
from pymilvus import MilvusClient, DataType
from FlagEmbedding import BGEM3FlagModel
from scipy import sparse as sp_sparse

TRANSLATED_DIR = PROJECT_ROOT / "data" / "translated"
SPARSE_DIM = 250002

NEW_DOMAINS = [
    "financial_markets", "fintech", "insurance", "investment_banking",
    "personal_finance", "risk_management", "stock_market",
]

# 原有 3 领域：仅允许写入 MySQL 训练表（Milvus 已有大量向量，严禁经此脚本重写）
ORIG_DOMAINS = ["banking", "corporate_finance", "financial_accounting"]

# MySQL 支持全部 10 领域；Milvus 仅支持 7 个新领域
ALL_DOMAINS = NEW_DOMAINS + ORIG_DOMAINS


# ─────────────────────────────────────────────────────────────────────────────
# (1) MySQL：高频问答样本
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_finance_faq():
    conn = pymysql.connect(
        host=_cfg.mysql.host, port=_cfg.mysql.port, user=_cfg.mysql.user,
        password=_cfg.mysql.password, database=_cfg.mysql.database, charset=_cfg.mysql.charset,
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS finance_faq (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        category    VARCHAR(50)  NOT NULL,
        question    VARCHAR(1000) NOT NULL,
        answer      TEXT         NOT NULL,
        source      VARCHAR(200) DEFAULT '',
        type        VARCHAR(50)  DEFAULT '',
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        KEY idx_category (category),
        KEY idx_question (question(191))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""")
    conn.commit()
    conn.close()


def load_translated(domain: str) -> list[dict]:
    path = TRANSLATED_DIR / f"{domain}.jsonl"
    if not path.exists():
        logger.warning(f"翻译文件不存在，跳过 {domain}: {path}")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def insert_mysql_highfreq(domains: list[str], per_domain: int):
    logger.info("=" * 50)
    logger.info("Step 1: MySQL finance_faq 高频问答（增量、幂等）")
    logger.info("=" * 50)
    _ensure_finance_faq()
    conn = pymysql.connect(
        host=_cfg.mysql.host, port=_cfg.mysql.port, user=_cfg.mysql.user,
        password=_cfg.mysql.password, database=_cfg.mysql.database, charset=_cfg.mysql.charset,
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    for domain in domains:
        rows = load_translated(domain)
        if not rows:
            continue
        # 幂等：先删除该领域旧行，再插入本次样本
        cur.execute("DELETE FROM finance_faq WHERE category=%s", (domain,))
        sample = rows
        if per_domain and per_domain < len(rows):
            # 按 type 分层抽样，覆盖全部题型
            by_type: dict[str, list[dict]] = {}
            for r in rows:
                by_type.setdefault(r.get("type", ""), []).append(r)
            types = list(by_type.keys())
            per_type = max(1, per_domain // max(1, len(types)))
            sample = []
            for t in types:
                random.shuffle(by_type[t])
                sample.extend(by_type[t][:per_type])
            if len(sample) < per_domain:
                rest = [r for r in rows if r not in sample]
                random.shuffle(rest)
                sample.extend(rest[: per_domain - len(sample)])
            random.shuffle(sample)
            sample = sample[:per_domain]
        for r in sample:
            cur.execute(
                """INSERT INTO finance_faq (category, question, answer, source, type)
                   VALUES (%s, %s, %s, %s, %s)""",
                (domain, r["question"], r["answer"], domain, r.get("type", "")),
            )
        logger.info(f"  {domain}: 写入 {len(sample)} 条高频问答")
    conn.commit()
    cur.execute("SELECT category, COUNT(*) AS cnt FROM finance_faq GROUP BY category ORDER BY category")
    dist = cur.fetchall()
    conn.close()
    logger.info("finance_faq 当前分布：")
    for row in dist:
        logger.info(f"  {row['category']:22s} : {row['cnt']} 条")


# ─────────────────────────────────────────────────────────────────────────────
# (2) Milvus：增量向量写入
# ─────────────────────────────────────────────────────────────────────────────
def _build_schema(client: MilvusClient):
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field("id",           DataType.INT64,   is_primary=True, auto_id=True)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR,     dim=1024)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("text",         DataType.VARCHAR,  max_length=65535)
    schema.add_field("category",     DataType.VARCHAR,  max_length=50)
    schema.add_field("type",         DataType.VARCHAR,  max_length=50)
    schema.add_field("question",     DataType.VARCHAR,  max_length=1000)
    schema.add_field("answer",       DataType.VARCHAR,  max_length=65535)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"nprobe": 16})
    return schema, index_params


def insert_milvus(domains: list[str], batch_size: int):
    logger.info("=" * 50)
    logger.info("Step 2: Milvus finrag_faq 增量向量写入")
    logger.info("=" * 50)

    milvus_kwargs = {"uri": _cfg.milvus.uri, "db_name": _cfg.milvus.database_name}
    if _cfg.milvus.token:
        milvus_kwargs["token"] = _cfg.milvus.token
    client = MilvusClient(**milvus_kwargs)

    # 集合不存在则创建（沿用既定 schema）
    if not client.has_collection(_cfg.milvus.collection_name):
        schema, index_params = _build_schema(client)
        client.create_collection(_cfg.milvus.collection_name, schema=schema, index_params=index_params)
        logger.info(f"集合 {_cfg.milvus.collection_name} 已新建")
    else:
        logger.info(f"集合 {_cfg.milvus.collection_name} 已存在，执行增量写入（保留原有数据）")

    # 幂等：删除本次涉及领域已有的旧向量，避免重复
    for domain in domains:
        try:
            client.delete(_cfg.milvus.collection_name, filter=f'category == "{domain}"',
                          consistency_level="Strong")
            logger.info(f"  已清理 {domain} 旧向量（如存在）")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  清理 {domain} 旧向量失败（可忽略，首次写入）: {e}")

    # 加载 BGE-M3 生成向量
    logger.info("加载 BGE-M3 模型 ...")
    model = BGEM3FlagModel(str(Path(_cfg.bge_m3_path)), use_fp16=False)
    logger.info("模型加载完成")

    total_inserted = 0
    for domain in domains:
        rows = load_translated(domain)
        if not rows:
            continue
        texts = [f"{r['question']} {r['answer']}" for r in rows]
        all_dense, all_sparse = [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = model.encode(batch, batch_size=batch_size, return_dense=True,
                                   return_sparse=True, return_colbert_vecs=False)
            all_dense.extend(result["dense_vecs"].tolist())
            for lexical in result["lexical_weights"]:
                if lexical:
                    indices = [int(k) for k in lexical.keys()]
                    values = [float(v) for v in lexical.values()]
                    max_idx = max(indices) if indices else 0
                    all_sparse.append(sp_sparse.csr_matrix(
                        (values, ([0] * len(indices), indices)), shape=(1, max_idx + 1)))
                else:
                    all_sparse.append(sp_sparse.csr_matrix((1, SPARSE_DIM)))
            logger.info(f"  [{domain}] 已向量化 {min(i + batch_size, len(texts))}/{len(texts)}")
        # 分批写入
        for i in range(0, len(rows), batch_size):
            batch_rows = [{
                "dense_vector":  all_dense[i + j],
                "sparse_vector": all_sparse[i + j],
                "text":     f"{rows[i + j]['question']}\n答：{rows[i + j]['answer']}",
                "category": rows[i + j]["domain"],
                "type":     rows[i + j].get("type", ""),
                "question": rows[i + j]["question"],
                "answer":   rows[i + j]["answer"],
            } for j in range(min(batch_size, len(rows) - i))]
            client.insert(collection_name=_cfg.milvus.collection_name, data=batch_rows)
            total_inserted += len(batch_rows)
            logger.info(f"  [{domain}] 已写入向量 {min(i + batch_size, len(rows))}/{len(rows)}")
        logger.info(f"  [{domain}] 完成，{len(rows)} 条")

    # 重建索引并加载
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"nprobe": 16})
    try:
        client.create_index(collection_name=_cfg.milvus.collection_name, index_params=index_params)
    except Exception as e:  # noqa: BLE001
        logger.info(f"  索引已存在，跳过重建: {e}")
    client.load_collection(_cfg.milvus.collection_name)
    stats = client.get_collection_stats(_cfg.milvus.collection_name)
    logger.info(f"Milvus 写入完成，本批新增 {total_inserted} 条，集合总行数 {stats.get('row_count')}")


def main():
    ap = argparse.ArgumentParser(description="新领域数据入库（MySQL 高频 + Milvus 向量）")
    ap.add_argument("--domain", type=str, default="", help="只处理指定领域，逗号分隔（默认全部新领域）")
    ap.add_argument("--mysql-per", type=int, default=40, help="MySQL 每领域高频样本数（0=全量翻译数据）")
    ap.add_argument("--milvus-batch", type=int, default=64, help="Milvus 向量化批次")
    ap.add_argument("--skip-milvus", action="store_true", help="跳过 Milvus，仅写 MySQL（用于训练样本平衡）")
    args = ap.parse_args()

    domains = args.domain.split(",") if args.domain else list(NEW_DOMAINS)
    domains = [d.strip() for d in domains if d.strip() in ALL_DOMAINS]
    if not domains:
        logger.error("没有可处理的领域")
        return

    # Milvus 仅处理新领域，防止误删原有 3 领域的大量存量向量
    milvus_domains = [d for d in domains if d in NEW_DOMAINS]

    logger.info("=" * 60)
    logger.info("新领域数据入库")
    logger.info(f"  领域: {domains}")
    logger.info(f"  MySQL 每领域高频样本: {args.mysql_per}")
    logger.info("=" * 60)

    insert_mysql_highfreq(domains, args.mysql_per)
    if args.skip_milvus:
        logger.info("--skip-milvus 已指定，跳过 Milvus 写入")
    elif milvus_domains:
        insert_milvus(milvus_domains, args.milvus_batch)
    else:
        logger.info("本次仅涉及原有领域，跳过 Milvus（保护存量向量）")

    logger.info("=" * 60)
    logger.info("全部入库完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
