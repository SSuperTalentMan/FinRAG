#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/db.py — FinRag MySQL 的异步适配层（%s 占位符 + asyncio.to_thread）。

FinRag 的 db/mysql.get_db() 是同步 pymysql；审查流水线运行于 async 事件循环，
为避免阻塞 SSE 并发，把阻塞 DB 调用压进线程池。表结构在此幂等建表（复用 FinRag
"启动时建表"约定，兼容未跑 init 脚本的存量库）。
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from db.mysql import get_db

_COMPLIANCE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS docaudit_document (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        doc_name VARCHAR(256) NOT NULL,
        doc_type VARCHAR(32) DEFAULT 'pdf',
        file_path VARCHAR(512) NOT NULL,
        file_hash CHAR(64) NOT NULL,
        page_count INT DEFAULT 0,
        parser_summary JSON NULL,
        created_by VARCHAR(64) DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_file_hash (file_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS docaudit_page_unit (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        document_id BIGINT NOT NULL,
        page_no INT NOT NULL,
        parser_source VARCHAR(32) DEFAULT 'pymupdf',
        route VARCHAR(16) DEFAULT 'digital',
        ocr_confidence FLOAT NULL,
        text MEDIUMTEXT NOT NULL,
        KEY idx_doc (document_id, page_no)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS docaudit_clause (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        document_id BIGINT NOT NULL,
        clause_no VARCHAR(64) NOT NULL,
        title VARCHAR(256) DEFAULT '',
        content MEDIUMTEXT NOT NULL,
        page_start INT DEFAULT 1,
        page_end INT DEFAULT 1,
        extract_status VARCHAR(16) DEFAULT 'ok',
        defects VARCHAR(500) DEFAULT '',
        KEY idx_doc (document_id),
        UNIQUE KEY uk_doc_clause (document_id, clause_no)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS docaudit_risk_rule (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        rule_id VARCHAR(64) NOT NULL,
        doc_name VARCHAR(256) DEFAULT '',
        article_no VARCHAR(64) DEFAULT '',
        requirement MEDIUMTEXT NOT NULL,
        severity VARCHAR(16) DEFAULT 'medium',
        keywords VARCHAR(500) DEFAULT '',
        UNIQUE KEY uk_rule_id (rule_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS docaudit_review_task (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        document_id BIGINT NOT NULL,
        stage VARCHAR(32) DEFAULT 'uploaded',
        progress JSON NULL,
        error VARCHAR(800) DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        KEY idx_doc (document_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS docaudit_clause_review (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        task_id BIGINT NOT NULL,
        clause_id BIGINT NOT NULL,
        clause_no VARCHAR(64) NOT NULL,
        verdict VARCHAR(32) NOT NULL,
        risk_level VARCHAR(16) DEFAULT 'low',
        violated_rules JSON NULL,
        evidence TEXT,
        suggestion TEXT,
        hitl_status VARCHAR(24),
        reviewer VARCHAR(64),
        reviewed_at TIMESTAMP NULL DEFAULT NULL,
        UNIQUE KEY uk_task_clause (task_id, clause_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS docaudit_report (
        task_id BIGINT PRIMARY KEY,
        summary JSON NULL,
        md_path VARCHAR(512) DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


def _sync(fn):
    return asyncio.to_thread(fn)


def ensure_tables() -> None:
    """幂等建表，应用启动/初始化时调用。"""
    with get_db() as conn:
        cur = conn.cursor()
        for ddl in _COMPLIANCE_DDL:
            cur.execute(ddl)
    logger.info("docaudit 审查相关表已就绪")


def fetch_all(sql: str, params: tuple | list | None = None) -> list[dict[str, Any]]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        return cur.fetchall()


def fetch_one(sql: str, params: tuple | list | None = None) -> dict[str, Any] | None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        return cur.fetchone()


def execute(sql: str, params: tuple | list | None = None) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        # list-of-sequences = 批量插入（pymysql execute 不支持，需 executemany）
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], (list, tuple)):
            cur.executemany(sql, params)
            return cur.rowcount
        cur.execute(sql, params or ())
        return cur.rowcount


def insert_returning_id(sql: str, params: tuple | list | None = None) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        return int(cur.lastrowid)


# ─── 异步包装（供审查流水线在 async 上下文调用）────────────────────────────────
async def afetch_all(sql: str, params: tuple | list | None = None) -> list[dict[str, Any]]:
    return await asyncio.to_thread(fetch_all, sql, params)


async def afetch_one(sql: str, params: tuple | list | None = None) -> dict[str, Any] | None:
    return await asyncio.to_thread(fetch_one, sql, params)


async def aexecute(sql: str, params: tuple | list | None = None) -> int:
    return await asyncio.to_thread(execute, sql, params)


async def ainsert_returning_id(sql: str, params: tuple | list | None = None) -> int:
    return await asyncio.to_thread(insert_returning_id, sql, params)