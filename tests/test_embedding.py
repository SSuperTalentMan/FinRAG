#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_embedding.py — BGE-M3 嵌入服务单元测试
"""
import pytest
from services.embedding import get_embedding_model, encode_query, encode_batch, encode_query_sparse


class TestEmbedding:
    def test_get_model_singleton(self):
        m1 = get_embedding_model()
        m2 = get_embedding_model()
        assert m1 is m2

    def test_encode_query_dim(self):
        vec = encode_query("银行贷款")
        assert isinstance(vec, list)
        assert len(vec) == 1024, f"期望1024维，实际{len(vec)}维"

    def test_encode_query_normalized(self):
        """归一化向量的模长应为 1。"""
        vec = encode_query("测试文本")
        import math
        norm = math.sqrt(sum(v * v for v in vec))
        assert 0.99 < norm < 1.01, f"归一化失败，模长={norm}"

    def test_encode_batch(self):
        texts = ["银行贷款", "并购重组", "资产负债表"]
        batches = encode_batch(texts)
        assert len(batches) == 3
        for vec in batches:
            assert len(vec) == 1024

    def test_encode_consistency(self):
        """相同文本应产生相同向量。"""
        v1 = encode_query("同一文本")
        v2 = encode_query("同一文本")
        assert v1 == v2

    def test_encode_query_sparse(self):
        """稀疏向量编码应返回非空结果。"""
        from scipy import sparse
        lexical, sparse_vec = encode_query_sparse("什么是活期存款?")
        assert isinstance(sparse_vec, sparse.csr_matrix)
        assert sparse_vec.shape[0] == 1
        assert sparse_vec.nnz > 0, "稀疏向量应包含非零元素"
