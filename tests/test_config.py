#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_config.py — 配置模块单元测试
"""
import pytest
from config import Config, get_config, reload_config


class TestConfig:
    def test_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reload(self):
        c1 = get_config()
        c2 = reload_config()
        assert c1 is not c2

    def test_mysql_config(self):
        cfg = get_config()
        assert cfg.mysql.host == "localhost"
        assert cfg.mysql.port == 3306
        assert cfg.mysql.database == "finrag_qa"
        # charset 来自 config.ini，若无则使用默认值
        assert cfg.mysql.charset in ("utf8mb4", "")

    def test_redis_config(self):
        cfg = get_config()
        assert cfg.redis.host == "localhost"
        assert cfg.redis.port == 6379
        assert cfg.redis.password == "1234"
        assert cfg.redis.db == 0

    def test_milvus_config(self):
        cfg = get_config()
        assert "localhost" in cfg.milvus.uri
        assert cfg.milvus.database_name == "finrag"
        assert cfg.milvus.collection_name == "finrag_faq"

    def test_llm_config(self):
        cfg = get_config()
        # P2 融合后统一为 qwen3.7-flash（用户指定，阿里云百炼）
        assert cfg.llm.model == "qwen3.7-flash"
        assert cfg.llm.temperature == 0.7
        assert cfg.llm.max_tokens == 2048
        assert len(cfg.llm.api_key) > 0

    def test_retrieval_config(self):
        cfg = get_config()
        assert cfg.retrieval.retrieval_k == 5
        assert cfg.retrieval.similarity_threshold == 0.7
        assert cfg.retrieval.parent_chunk_size == 1200

    def test_app_config(self):
        cfg = get_config()
        assert cfg.app.host == "0.0.0.0"
        assert cfg.app.port == 8000

    def test_bge_paths(self):
        cfg = get_config()
        assert cfg.bge_m3_path.endswith("bge-m3")
        assert cfg.bge_reranker_path.endswith("bge-reranker-v2-m3")
        assert cfg.bge_m3_dim == 1024

    def test_api_key_from_env(self, monkeypatch):
        """API Key 优先从环境变量读取。"""
        monkeypatch.setenv("API_KEY", "test-key-from-env")
        cfg = reload_config()
        assert cfg.api_key == "test-key-from-env"
        # 清理
        monkeypatch.delenv("API_KEY", raising=False)
        reload_config()
