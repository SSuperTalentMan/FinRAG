#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_redis_db.py — Redis 缓存单元测试（全 mock，无需真实 Redis）
"""
import pytest
from unittest.mock import patch, MagicMock
from db.redis import get_redis, make_cache_key, cache_get, cache_set, cache_delete


class TestRedis:
    def test_make_cache_key_deterministic(self):
        key1 = make_cache_key("测试文本")
        key2 = make_cache_key("测试文本")
        assert key1 == key2
        assert key1.startswith("finrag:qa:")

    def test_make_cache_key_different_text(self):
        key1 = make_cache_key("文本A")
        key2 = make_cache_key("文本B")
        assert key1 != key2

    @patch("db.redis.get_redis")
    def test_cache_roundtrip(self, mock_get_redis):
        mock_r = MagicMock()
        mock_r.get.return_value = None
        mock_get_redis.return_value = mock_r

        test_text = "pytest_cache_test_xyz"
        test_data = {"answer": "测试答案", "domain": "banking"}
        cache_set(test_text, test_data, ttl=60)
        mock_r.setex.assert_called_once()
        assert mock_r.setex.call_args[0][0] == f"finrag:qa:{make_cache_key(test_text)[10:]}"

    @patch("db.redis.get_redis")
    def test_cache_miss(self, mock_get_redis):
        mock_r = MagicMock()
        mock_r.get.return_value = None
        mock_get_redis.return_value = mock_r

        result = cache_get("nonexistent_key_xyz_12345")
        assert result is None

    @patch("db.redis.get_redis")
    def test_cache_get_with_hit(self, mock_get_redis):
        import json
        test_data = {"answer": "hello", "domain": "general"}
        mock_r = MagicMock()
        mock_r.get.return_value = json.dumps(test_data)
        mock_get_redis.return_value = mock_r

        result = cache_get("some_query")
        assert result == test_data

    @patch("db.redis.get_redis")
    def test_cache_delete(self, mock_get_redis):
        mock_r = MagicMock()
        mock_get_redis.return_value = mock_r
        cache_delete("some_query")
        mock_r.delete.assert_called_once()

    @patch("db.redis.get_redis")
    def test_get_redis_connection(self, mock_get_redis):
        mock_r = MagicMock()
        mock_r.ping.return_value = True
        mock_get_redis.return_value = mock_r
        r = get_redis()
        assert r.ping() is True
