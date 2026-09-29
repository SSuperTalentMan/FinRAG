#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_mysql_db.py — MySQL 数据库操作单元测试（全 mock，无需真实 MySQL）
db/mysql.py 的 get_db() 走 dbutils 连接池（_get_pool().connection()），
因此本测试 patch _get_pool 而非 pymysql.connect，否则 mock 不会生效。
"""
import pytest
from unittest.mock import patch, MagicMock
from db.mysql import get_db, get_all_qa_for_bm25, get_qa_by_category, search_qa_exact


def _make_cursor(rows):
    """创建带过滤能力的 mock cursor。"""
    cur = MagicMock()
    cur._data = list(rows)

    def fake_execute(sql, params=None):
        if params and isinstance(params, tuple) and params[0] in ("banking", "corporate_finance", "financial_accounting"):
            cur._data = [r for r in rows if r.get("category") == params[0]]
        else:
            cur._data = list(rows)

    def fake_fetchall():
        return cur._data

    # fetchone 默认返回 None（模拟查不到数据）
    cur.execute = fake_execute
    cur.fetchall = fake_fetchall
    cur.fetchone.return_value = None
    cur.lastrowid = 1
    return cur


def _make_conn(rows=None):
    if rows is None:
        rows = [
            {"id": 1, "category": "banking", "question": "银行贷款年利率是多少",
             "answer": "银行贷款年利率根据政策而定", "source": "mysql", "type": "事实性"},
            {"id": 2, "category": "corporate_finance", "question": "公司并购估值方法",
             "answer": "常用DCF和可比公司法", "source": "mysql", "type": "推理型"},
            {"id": 3, "category": "financial_accounting", "question": "资产负债表编制方法",
             "answer": "按会计准则编制", "source": "mysql", "type": "事实性"},
        ]
    cur = _make_cursor(rows)
    conn = MagicMock()
    # 关键：用 spec 固定 cursor 返回同一个对象，避免 MagicMock 每次创建新实例
    conn.cursor = MagicMock(return_value=cur)
    conn.commit = MagicMock(return_value=None)
    return conn


def _make_context_conn(rows=None):
    """创建支持上下文管理协议的 mock 连接（用于 get_db 的 with 语句）。"""
    conn = _make_conn(rows)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


def _make_pool(rows=None):
    """创建 mock 连接池：pool.connection() 返回 mock 连接，供 get_db() 使用。"""
    pool = MagicMock()
    pool.connection.return_value = _make_context_conn(rows)
    return pool


# ─── 测试类 ────────────────────────────────────────────────────────────────────
class TestMySQL:
    @patch("db.mysql._get_pool", return_value=_make_pool())
    def test_get_all_qa_for_bm25(self, mock_pool):
        items = get_all_qa_for_bm25()
        assert isinstance(items, list)
        assert len(items) == 3
        assert items[0]["question"] == "银行贷款年利率是多少"
        assert items[0]["category"] == "banking"

    @patch("db.mysql._get_pool", return_value=_make_pool())
    def test_get_qa_by_category(self, mock_pool):
        items = get_qa_by_category("banking", limit=10)
        assert isinstance(items, list)
        assert len(items) > 0
        assert all(item.category == "banking" for item in items)

    @patch("db.mysql._get_pool")
    def test_search_qa_exact_not_found(self, mock_get_pool):
        mock_conn = _make_context_conn()
        mock_conn.cursor.return_value.fetchone.return_value = None
        mock_get_pool.return_value.connection.return_value = mock_conn
        result = search_qa_exact("this_question_does_not_exist_xyz123")
        assert result is None

    @patch("db.mysql._get_pool", return_value=_make_pool())
    def test_get_db_context_manager(self, mock_pool):
        with get_db() as conn:
            assert conn is not None
            cur = conn.cursor()
            cur.execute("SELECT 1 AS val")
            # 通过 _make_cursor 内部设置 fetchone 返回值
            mock_row = {"val": 1}
            cur.fetchone.return_value = mock_row
            row = cur.fetchone()
            assert row is not None
            assert row["val"] == 1
