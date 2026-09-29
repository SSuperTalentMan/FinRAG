#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/embedding.py — BGE-M3 嵌入服务
负责将文本转换为 BGE-M3 的稠密向量（dense）和稀疏向量（sparse）。
稠密向量：1024 维归一化向量，用于向量相似度检索。
稀疏向量：词项权重向量（lexical weights），用于混合稀疏检索。
"""

from typing import Tuple
import os
import threading

import torch
from FlagEmbedding import BGEM3FlagModel
from scipy import sparse as sp_sparse
from loguru import logger
from config import get_config

_model: BGEM3FlagModel | None = None
_model_lock = threading.Lock()  # 双检锁：避免并发首请求重复加载模型（内存/CPU 开销大）
_BATCH_SIZE = 64


def _limit_cpu_threads() -> None:
    """
    限制 PyTorch CPU 线程数，避免文档批量编码时吃满所有核心导致整机卡顿。
    默认用一半核心（最少 2），可用环境变量 FINRAG_TORCH_THREADS 覆盖。
    """
    total = os.cpu_count() or 4
    try:
        n = int(os.environ.get("FINRAG_TORCH_THREADS", max(2, total // 2)))
    except ValueError:
        n = max(2, total // 2)
    try:
        torch.set_num_threads(n)
        logger.info(f"PyTorch CPU 线程数限制: {n}/{total}")
    except Exception as e:
        logger.warning(f"设置 PyTorch 线程数失败: {e}")


def get_embedding_model() -> BGEM3FlagModel:
    """懒加载 BGE-M3 模型（单例，线程安全）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                cfg = get_config()
                _limit_cpu_threads()
                logger.info(f"加载 BGE-M3 模型: {cfg.bge_m3_path}")
                _model = BGEM3FlagModel(cfg.bge_m3_path, use_fp16=False)
                logger.info("BGE-M3 模型加载完成 (dense=1024dim, sparse=lexical)")
    return _model


def is_model_loaded() -> bool:
    """BGE-M3 模型是否已加载（供健康检查只读探测，不触发加载）。"""
    return _model is not None


def encode_query(text: str) -> list[float]:
    """对单个查询文本编码为 1024 维归一化稠密向量。"""
    model = get_embedding_model()
    result = model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return result["dense_vecs"][0].tolist()


def encode_query_dense_sparse(text: str) -> Tuple[list[float], sp_sparse.csr_matrix]:
    """
    对单个查询文本同时编码为稠密向量和稀疏向量。
    返回 (dense_vec, sparse_vec)，其中 sparse_vec 为 scipy CSR 矩阵 (1, vocab_size)。
    """
    model = get_embedding_model()
    result = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vec = result["dense_vecs"][0].tolist()

    # 将 lexical_weights 转换为 scipy CSR 稀疏矩阵
    lexical = result["lexical_weights"][0]
    if lexical:
        indices = [int(k) for k in lexical.keys()]
        values = [float(v) for v in lexical.values()]
        vocab_size = max(indices) + 1 if indices else 250002
        sparse_vec = sp_sparse.csr_matrix(
            (values, ([0] * len(indices), indices)),
            shape=(1, vocab_size),
        )
    else:
        sparse_vec = sp_sparse.csr_matrix((1, 250002))

    return dense_vec, sparse_vec


def encode_batch(texts: list[str]) -> list[list[float]]:
    """批量编码文本列表，返回归一化稠密向量列表。"""
    model = get_embedding_model()
    result = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return result["dense_vecs"].tolist()


def encode_query_sparse(text: str) -> Tuple[dict, sp_sparse.csr_matrix]:
    """
    对单个查询文本编码为稀疏向量（lexical weights）。
    返回 (token_dict, sparse_matrix)，token_dict 为 {token_id_str: weight}。
    """
    model = get_embedding_model()
    result = model.encode(
        [text],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    lexical = result["lexical_weights"][0]
    # 转换为 scipy CSR 矩阵
    if lexical:
        indices = [int(k) for k in lexical.keys()]
        values = [float(v) for v in lexical.values()]
        vocab_size = max(indices) + 1 if indices else 250002
        sparse_vec = sp_sparse.csr_matrix(
            (values, ([0] * len(indices), indices)),
            shape=(1, vocab_size),
        )
    else:
        sparse_vec = sp_sparse.csr_matrix((1, 250002))
    return lexical, sparse_vec


def encode_batch_dense_sparse(
    texts: list[str],
) -> Tuple[list[list[float]], list[sp_sparse.csr_matrix]]:
    """
    批量编码文本列表，同时返回稠密向量和稀疏向量。
    返回 (dense_list, sparse_list)，每对对应同一文本。
    """
    model = get_embedding_model()
    # 注意：BGEM3FlagModel.encode 不支持 show_progress_bar 参数，
    # 传入会被透传给 tokenizer.pad() 导致 TypeError（PreTrainedTokenizerBase.pad()
    # got an unexpected keyword argument 'show_progress_bar'），故不可传。
    result = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_list = result["dense_vecs"].tolist()

    sparse_list = []
    for lexical in result["lexical_weights"]:
        if lexical:
            indices = [int(k) for k in lexical.keys()]
            values = [float(v) for v in lexical.values()]
            vocab_size = max(indices) + 1 if indices else 250002
            sparse_list.append(
                sp_sparse.csr_matrix(
                    (values, ([0] * len(indices), indices)),
                    shape=(1, vocab_size),
                )
            )
        else:
            sparse_list.append(sp_sparse.csr_matrix((1, 250002)))

    return dense_list, sparse_list
