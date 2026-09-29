#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""db/__init__.py — 数据库模块包导出"""
from .mysql import get_db, get_qa_by_category, get_all_qa_for_bm25, search_qa_exact
from .redis import get_redis, make_cache_key, cache_get, cache_set, cache_delete
from .milvus import get_milvus_client, search_milvus, ensure_collection, COLLECTION_NAME

__all__ = [
    "get_db", "get_qa_by_category", "get_all_qa_for_bm25", "search_qa_exact",
    "get_redis", "make_cache_key", "cache_get", "cache_set", "cache_delete",
    "get_milvus_client", "search_milvus", "ensure_collection", "COLLECTION_NAME",
]
