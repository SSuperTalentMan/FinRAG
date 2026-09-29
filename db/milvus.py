#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
db/milvus.py — Milvus 向量数据库操作（BGE-M3 稠密+稀疏混合向量）
支持：
  - dense_vector：1024 维稠密向量（IVF_FLAT 索引，IP 度量）
  - sparse_vector：词项稀疏向量（SPARSE_INVERTED_INDEX 索引，IP 度量）
"""

from typing import Tuple
import os
import re
import threading

from pymilvus import MilvusClient, DataType
from scipy import sparse as sp_sparse
from loguru import logger
from config import get_config

# ─── Collection 常量 ───────────────────────────────────────────────────────────
COLLECTION_NAME = "finrag_faq"

# domain/category 白名单：仅允许字母/数字/下划线，且必须在 valid_sources 中。
# 防止 filter 表达式注入（如 domain='x" or 1==1 //' 窥探其他知识库向量）。
_DOMAIN_RE = re.compile(r"^[a-zA-Z0-9_]+$")


def _safe_domain_filter(domain: str | None) -> str:
    """构造安全的 Milvus domain 过滤表达式。

    domain 为 None/空/非法时返回空串（不过滤）；
    仅当 domain 匹配白名单正则（字母+数字+下划线）时才拼入 filter，
    杜绝引号/操作符注入。
    """
    if not domain:
        return ""
    if not _DOMAIN_RE.match(domain):
        logger.warning(f"domain 值非法，已忽略过滤: {domain!r}")
        return ""
    return f'category == "{domain}"'

# 稀疏向量维度（BGE-M3 tokenizer vocab_size）
SPARSE_VEC_DIM = 250002

# 进程级单例客户端（MilvusClient 内部维护连接，避免每次请求都新建连接）
_milvus_client: MilvusClient | None = None

# collection 就绪标志：避免每次检索都做一次 has_collection/describe 往返。
# 仅在首检通过后置位；任一步骤失败不置位，下次调用自动重试。
_collection_ready = False
_collection_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端（Singleton）。"""
    global _milvus_client
    if _milvus_client is None:
        cfg = get_config()
        # 本地 standalone 默认无鉴权时传空 token 会让 SDK 跳过认证，避免 "authentication failed"
        kwargs = {"uri": cfg.milvus.uri, "db_name": cfg.milvus.database_name}
        if cfg.milvus.token:
            kwargs["token"] = cfg.milvus.token
        _milvus_client = MilvusClient(**kwargs)
        logger.info(f"Milvus 单例客户端已初始化: {cfg.milvus.uri} db={cfg.milvus.database_name}")
    return _milvus_client


def close_milvus() -> None:
    """应用关闭时释放 Milvus 连接（优雅停机）。"""
    global _milvus_client
    if _milvus_client is not None:
        try:
            _milvus_client.close()
        except Exception as e:
            logger.warning(f"关闭 Milvus 连接失败: {e}")
        _milvus_client = None


def ensure_collection(client: MilvusClient) -> None:
    """确保 collection 存在且 schema 正确（dense_vector + sparse_vector 双向量）。

    schema 不匹配时绝不静默 drop（会清空全部已入库向量）：
    - 默认抛 RuntimeError 并给出处置指引（人工确认后重建 + 重新导入数据）；
    - 开发环境可设 MILVUS_AUTO_RECREATE=1 恢复旧的自动重建行为（会清空数据）。
    """
    global _collection_ready
    if _collection_ready:
        return
    with _collection_lock:
        if _collection_ready:
            return
        if client.has_collection(COLLECTION_NAME):
            logger.info(f"Collection '{COLLECTION_NAME}' 已存在，检查 schema...")
            coll = client.describe_collection(COLLECTION_NAME)
            field_names = {f["name"] for f in coll["fields"]}
            # 检查是否已有双向量 schema
            if "dense_vector" in field_names and "sparse_vector" in field_names:
                logger.info("Collection schema 验证通过 (dense+sparse)")
                _collection_ready = True
                return
            # schema 不匹配：生产环境拒绝自动删数据
            mismatch_msg = (
                f"Collection '{COLLECTION_NAME}' schema 不匹配 (fields={field_names})，"
                f"缺少双向量字段。请人工确认后：1) 运行 scripts/recreate_collection.py --yes 重建并重新导入数据；"
                f"或 2) 开发环境设置 MILVUS_AUTO_RECREATE=1 自动重建（会清空全部向量数据）。"
            )
            logger.error(mismatch_msg)
            if os.getenv("MILVUS_AUTO_RECREATE", "").lower() in ("1", "true", "yes"):
                logger.warning("MILVUS_AUTO_RECREATE=1，drop 并按最新 schema 重建 collection（数据将清空）")
                client.drop_collection(COLLECTION_NAME)
            else:
                raise RuntimeError(mismatch_msg)

        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("id",          DataType.INT64,  is_primary=True, auto_id=True)
        schema.add_field("dense_vector",DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("text",        DataType.VARCHAR, max_length=65535)
        schema.add_field("category",    DataType.VARCHAR, max_length=50)
        schema.add_field("type",        DataType.VARCHAR, max_length=50)
        schema.add_field("question",    DataType.VARCHAR, max_length=1000)
        schema.add_field("answer",      DataType.VARCHAR, max_length=65535)

        index_params = client.prepare_index_params()
        # 稠密向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        # 稀疏向量索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"nprobe": 16},
        )

        client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
        _collection_ready = True
        logger.info(f"Collection '{COLLECTION_NAME}' 创建完成 (dense=1024, sparse=lexical)")


def search_milvus(
    client: MilvusClient,
    query_dense: list[float],
    query_sparse: sp_sparse.csr_matrix,
    k: int = 5,
    domain: str | None = None,
) -> list[dict]:
    """
    在 Milvus 中执行稠密+稀疏混合检索。
    分别搜索 dense 和 sparse 向量，按 question 去重合并。
    若指定 domain，则通过 filter 过滤。
    返回 [ {"score", "question", "answer", "category", "type", "source"} ]
    """
    filter_expr = _safe_domain_filter(domain)

    # 稠密向量检索
    dense_results = _search_dense(client, query_dense, k, filter_expr)
    # 稀疏向量检索
    sparse_results = _search_sparse(client, query_sparse, k, filter_expr)

    # 按 question 去重合并（dense 优先）
    merged: dict[str, dict] = {}
    for r in dense_results:
        merged[r["question"]] = r
    for r in sparse_results:
        if r["question"] not in merged:
            r["source"] = "milvus_sparse"
            merged[r["question"]] = r

    results = list(merged.values())
    logger.debug(f"混合检索完成: dense={len(dense_results)}, sparse={len(sparse_results)}, merged={len(results)}")
    return results


def _search_dense(
    client: MilvusClient,
    query_dense: list[float],
    k: int,
    filter_expr: str,
) -> list[dict]:
    """稠密向量检索。"""
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_dense],
        limit=k,
        search_params={"metric_type": "IP", "params": {"nprobe": 16}},
        anns_field="dense_vector",
        filter=filter_expr,
        output_fields=["question", "answer", "category", "type", "text"],
    )
    results = []
    for hit_list in search_res:
        for hit in hit_list:
            results.append({
                "score":    round(float(hit["distance"]), 6),
                "question": hit["entity"]["question"],
                "answer":   hit["entity"]["answer"],
                # text 是子块原文，也是当初向量化时用的文本，重排必须与之一致
                "text":     hit["entity"].get("text", ""),
                "category": hit["entity"]["category"],
                "type":     hit["entity"].get("type", ""),
                "source":   "milvus_dense",
            })
    return results


def _search_sparse(
    client: MilvusClient,
    query_sparse: sp_sparse.csr_matrix,
    k: int,
    filter_expr: str,
) -> list[dict]:
    """稀疏向量检索。"""
    search_res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_sparse],
        limit=k,
        search_params={"metric_type": "IP", "params": {"nprobe": 8}},
        anns_field="sparse_vector",
        filter=filter_expr,
        output_fields=["question", "answer", "category", "type", "text"],
    )
    results = []
    for hit_list in search_res:
        for hit in hit_list:
            results.append({
                "score":    round(float(hit["distance"]), 6),
                "question": hit["entity"]["question"],
                "answer":   hit["entity"]["answer"],
                # text 是子块原文，也是当初向量化时用的文本，重排必须与之一致
                "text":     hit["entity"].get("text", ""),
                "category": hit["entity"]["category"],
                "type":     hit["entity"].get("type", ""),
                "source":   "milvus_sparse",
            })
    return results
