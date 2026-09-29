#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
init_data.py — 金融 RAG 数据初始化脚本（优化版：使用底层 transformers 加速）
功能：
  1. 清空并重建 MySQL 表，注入约 100 条高频问答对
  2. 清空并重建 Milvus collection，将全部 4,660 条本地数据用 BGE-M3 向量化后存入
     - dense_vector：1024 维稠密向量（IVF_FLAT 索引）
     - sparse_vector：词项稀疏向量（SPARSE_INVERTED_INDEX 索引）
依赖：pymysql, redis, pymilvus, transformers, scipy, loguru
运行：python init_data.py
"""

import os
import sys
import json
import random
from pathlib import Path
from loguru import logger

# 确保项目根目录在 sys.path（支持从任意目录运行）
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── CPU 性能优化（必须在 torch 导入前设置）──────────────────────────────
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["MKL_NUM_THREADS"]      = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

# 配置统一从 config.py（config.ini + 环境变量）读取，避免把路径/账号写死
from config import get_config
_cfg = get_config()

import torch
torch.set_num_threads(4)

import pymysql
from pymilvus import MilvusClient, DataType
from FlagEmbedding import BGEM3FlagModel
from scipy import sparse as sp_sparse

# ─── 路径常量（基于项目根目录，跨平台 / 容器通用）────────────────────────────
DATA_DIR     = PROJECT_ROOT / "data"
BGE_M3_PATH  = Path(_cfg.bge_m3_path)
LOG_FILE     = PROJECT_ROOT / "logs" / "init_data.log"

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
    str(LOG_FILE),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
)

# ─── 配置（来自 config.py，兼容环境变量覆盖）──────────────────────────────────
MYSQL_CONFIG = {
    "host": _cfg.mysql.host, "port": _cfg.mysql.port,
    "user": _cfg.mysql.user, "password": _cfg.mysql.password,
    "database": _cfg.mysql.database, "charset": _cfg.mysql.charset,
}
MILVUS_CONFIG = {
    "uri": _cfg.milvus.uri,
    "token": _cfg.milvus.token,
    "database_name": _cfg.milvus.database_name,
    "collection_name": _cfg.milvus.collection_name,
}
BATCH_SIZE = 64
SPARSE_DIM = 250002


def load_all_qa_pairs() -> list[dict]:
    logger.info("开始加载本地数据...")
    all_pairs = []
    domain_map = {
        "banking_data":            "banking",
        "Corporate_Finance_data":  "corporate_finance",
        "Financial_Accounting_data": "financial_accounting",
    }
    for subdir, domain in domain_map.items():
        dir_path = DATA_DIR / subdir
        if not dir_path.exists():
            logger.warning(f"目录不存在，跳过：{dir_path}")
            continue
        count = 0
        for fname in sorted(dir_path.glob("*.jsonl")):
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        all_pairs.append({
                            "domain": domain,
                            "question": item["question"],
                            "answer":   item["answer"],
                            "type":     item.get("type", ""),
                        })
                        count += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"跳过无效行 {fname.name}: {e}")
        logger.info(f"  {domain}: 加载 {count} 条")
    logger.info(f"共加载 {len(all_pairs)} 条问答对")
    return all_pairs


def stratified_sample(pairs: list[dict], total: int = 100) -> list[dict]:
    random.seed(42)
    by_domain: dict[str, list[dict]] = {}
    for p in pairs:
        by_domain.setdefault(p["domain"], []).append(p)
    domains = list(by_domain.keys())
    n_per_domain = total // len(domains)
    sampled = []
    for domain in domains:
        items = by_domain[domain]
        by_type: dict[str, list[dict]] = {}
        for item in items:
            by_type.setdefault(item["type"], []).append(item)
        types = list(by_type.keys())
        n_per_type = n_per_domain // len(types) if types else 0
        for t in types:
            subset = by_type[t]
            draw = min(n_per_type, len(subset))
            sampled.extend(random.sample(subset, draw))
        shortfall = n_per_domain - len(sampled)
        if shortfall > 0:
            remaining = [p for p in items if p not in sampled]
            sampled.extend(random.sample(remaining, min(shortfall, len(remaining))))
    logger.info(f"分层抽样完成，共 {len(sampled)} 条高频问答对")
    return sampled


def init_mysql(sampled: list[dict]) -> None:
    logger.info("=" * 50)
    logger.info("Step 1: 初始化 MySQL")
    logger.info("=" * 50)
    conn = pymysql.connect(**MYSQL_CONFIG)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS finance_faq")
    cur.execute("""
        CREATE TABLE finance_faq (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    logger.info("finance_faq 表已重建")
    inserted = 0
    for item in sampled:
        cur.execute(
            """INSERT INTO finance_faq (category, question, answer, source, type)
               VALUES (%s, %s, %s, %s, %s)""",
            (item["domain"], item["question"], item["answer"], item["domain"], item["type"]),
        )
        inserted += 1
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM finance_faq")
    logger.info(f"MySQL 插入完成，共 {cur.fetchone()[0]} 条")
    cur.execute("SELECT category, COUNT(*) as cnt FROM finance_faq GROUP BY category ORDER BY category")
    for row in cur.fetchall():
        logger.info(f"  {row[0]:25s} : {row[1]} 条")
    conn.close()


def build_milvus_collection(client: MilvusClient) -> None:
    logger.info("=" * 50)
    logger.info("Step 2: 初始化 Milvus Collection")
    logger.info("=" * 50)
    if client.has_collection(MILVUS_CONFIG["collection_name"]):
        client.drop_collection(MILVUS_CONFIG["collection_name"])
        logger.info(f"旧 collection '{MILVUS_CONFIG['collection_name']}' 已删除")

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

    client.create_collection(MILVUS_CONFIG["collection_name"], schema=schema, index_params=index_params)
    logger.info(f"Collection '{MILVUS_CONFIG['collection_name']}' 创建完成 (dense=1024, sparse=lexical)")


def embed_and_insert(client: MilvusClient, all_pairs: list[dict]) -> None:
    """使用 FlagEmbedding BGEM3FlagModel 对所有数据生成稠密+稀疏向量并批量写入 Milvus。"""
    logger.info("=" * 50)
    logger.info("Step 3: BGE-M3 向量化（稠密+稀疏）& 写入 Milvus")
    logger.info("=" * 50)

    logger.info("加载 BGE-M3 模型（FlagEmbedding）...")
    model = BGEM3FlagModel(str(BGE_M3_PATH), use_fp16=False)
    logger.info("模型加载完成")

    total = len(all_pairs)
    texts = [p["question"] + " " + p["answer"] for p in all_pairs]
    logger.info(f"共 {total} 条数据，开始向量化（稠密+稀疏）...")

    all_dense: list[list[float]] = []
    all_sparse: list[sp_sparse.csr_matrix] = []

    # 分批编码
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        result = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        all_dense.extend(result["dense_vecs"].tolist())

        for lexical in result["lexical_weights"]:
            if lexical:
                indices = [int(k) for k in lexical.keys()]
                values = [float(v) for v in lexical.values()]
                max_idx = max(indices) if indices else 0
                sparse_vec = sp_sparse.csr_matrix(
                    (values, ([0] * len(indices), indices)),
                    shape=(1, max_idx + 1),
                )
            else:
                sparse_vec = sp_sparse.csr_matrix((1, SPARSE_DIM))
            all_sparse.append(sparse_vec)

        logger.info(f"  已向量化 {min(i + BATCH_SIZE, len(texts))}/{total}")

    logger.info(f"向量化完成，开始写入 Milvus...")

    for i in range(0, total, BATCH_SIZE):
        batch_insert = [
            {
                "dense_vector":  all_dense[i + j],
                "sparse_vector": all_sparse[i + j],
                "text": all_pairs[i + j]["question"] + "\n答：" + all_pairs[i + j]["answer"],
                "category": all_pairs[i + j]["domain"],
                "type": all_pairs[i + j]["type"],
                "question": all_pairs[i + j]["question"],
                "answer": all_pairs[i + j]["answer"],
            }
            for j in range(min(BATCH_SIZE, total - i))
        ]
        client.insert(collection_name=MILVUS_CONFIG["collection_name"], data=batch_insert)
        logger.info(f"  已写入 {min(i + BATCH_SIZE, total)}/{total}")

    # 重建索引
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"nprobe": 16})
    client.create_index(collection_name=MILVUS_CONFIG["collection_name"], index_params=index_params)
    logger.info("索引构建完成")

    client.load_collection(MILVUS_CONFIG["collection_name"])
    # 通过 stats 获取准确行数
    stats = client.get_collection_stats(MILVUS_CONFIG["collection_name"])
    total_count = stats.get("row_count", 0)
    logger.info(f"Milvus 写入完成，共 {total_count} 条")


def verify(client: MilvusClient) -> None:
    logger.info("=" * 50)
    logger.info("Step 4: 验证数据")
    logger.info("=" * 50)

    conn = pymysql.connect(**MYSQL_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM finance_faq")
    logger.info(f"MySQL finance_faq 行数: {cur.fetchone()[0]}")
    cur.execute("SELECT category, COUNT(*) FROM finance_faq GROUP BY category")
    for r in cur.fetchall():
        logger.info(f"  {r[0]:25s} : {r[1]} 条")
    conn.close()

    stats = client.get_collection_stats(MILVUS_CONFIG["collection_name"])
    logger.info(f"Milvus finrag_faq 行数: {stats.get('row_count', 0)}")

    # 随机查询验证
    res = client.query(
        collection_name=MILVUS_CONFIG["collection_name"],
        filter="", limit=1,
        output_fields=["category", "question", "answer"],
    )
    if res:
        row = res[0]
        logger.info(f"示例数据: [{row['category']}] {row['question'][:50]}...")

    # 混合检索测试
    logger.info("执行混合检索测试...")
    test_q = all_pairs[0]["question"]
    test_result = model.encode(
        [test_q],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    test_dense = test_result["dense_vecs"][0].tolist()
    test_lexical = test_result["lexical_weights"][0]
    if test_lexical:
        indices = [int(k) for k in test_lexical.keys()]
        values = [float(v) for v in test_lexical.values()]
        test_sparse = sp_sparse.csr_matrix(
            (values, ([0] * len(indices), indices)),
            shape=(1, max(indices) + 1),
        )
    else:
        test_sparse = sp_sparse.csr_matrix((1, SPARSE_DIM))

    # 稠密检索
    dense_res = client.search(
        collection_name=MILVUS_CONFIG["collection_name"],
        data=[test_dense], limit=3,
        search_params={"metric_type": "IP", "params": {"nprobe": 16}},
        anns_field="dense_vector",
        output_fields=["category", "question"],
    )
    for i, hit in enumerate(dense_res[0]):
        logger.info(f"  稠密检索[{i}]: score={hit['distance']:.4f}  cat={hit['entity'].get('category','?')}")

    # 稀疏检索
    sparse_res = client.search(
        collection_name=MILVUS_CONFIG["collection_name"],
        data=[test_sparse], limit=3,
        search_params={"metric_type": "IP", "params": {"nprobe": 8}},
        anns_field="sparse_vector",
        output_fields=["category", "question"],
    )
    for i, hit in enumerate(sparse_res[0]):
        logger.info(f"  稀疏检索[{i}]: score={hit['distance']:.4f}  cat={hit['entity'].get('category','?')}")

    logger.info("=" * 50)
    logger.info("数据初始化全部完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    logger.info("FinRag 数据初始化脚本启动")
    logger.info(f"项目根目录: {PROJECT_ROOT}")

    all_pairs = load_all_qa_pairs()
    sampled   = stratified_sample(all_pairs, total=100)
    init_mysql(sampled)

    # Milvus standalone 默认无鉴权时传空 token 会让 SDK 跳过认证，避免报错
    milvus_kwargs = {
        "uri": MILVUS_CONFIG["uri"],
        "db_name": MILVUS_CONFIG["database_name"],
    }
    if MILVUS_CONFIG["token"]:
        milvus_kwargs["token"] = MILVUS_CONFIG["token"]
    client = MilvusClient(**milvus_kwargs)
    build_milvus_collection(client)
    embed_and_insert(client, all_pairs)
    verify(client)

    logger.info("脚本执行完毕")
