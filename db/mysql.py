#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
db/mysql.py — MySQL 连接池与 FAQ 查询
"""

from typing import Optional
import time

import pymysql
import bcrypt
from contextlib import contextmanager
from loguru import logger

try:
    from dbutils.pooled_db import PooledDB
    _HAS_POOL = True
except ImportError:  # 未安装连接池时降级为短连接
    _HAS_POOL = False

from config import get_config
from models import BM25Hit, UserInfo


# 进程级连接池（单例），避免每条请求新建连接，提升并发下的稳定性
_pool = None


def _get_pool():
    """懒初始化 MySQL 连接池（Singleton）。"""
    global _pool
    if _pool is None:
        cfg = get_config()
        _pool = PooledDB(
            creator=pymysql,
            maxconnections=20,        # 最大连接数，按 16G 机器与并发量调整
            mincached=2,               # 初始化时保持的空闲连接
            maxcached=8,               # 连接池最多空闲连接
            maxshared=0,
            blocking=True,             # 连接数耗尽时阻塞等待，而非报错
            ping=1,                    # 取连接时检查存活
            host=cfg.mysql.host,
            port=cfg.mysql.port,
            user=cfg.mysql.user,
            password=cfg.mysql.password,
            database=cfg.mysql.database,
            charset=cfg.mysql.charset,
            cursorclass=pymysql.cursors.DictCursor,
        )
        logger.info(f"MySQL 连接池已初始化: {cfg.mysql.host}:{cfg.mysql.port}/{cfg.mysql.database}")
    return _pool


def close_mysql_pool() -> None:
    """应用关闭时释放连接池（优雅停机）。"""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception as e:
            logger.warning(f"关闭 MySQL 连接池失败: {e}")
        _pool = None


@contextmanager
def get_db():
    """获取数据库连接的上下文管理器，自动提交/回滚/归还连接。

    优先复用连接池；若未安装 dbutils 则降级为短连接。
    """
    cfg = get_config()
    if _HAS_POOL:
        conn = _get_pool().connection()
    else:
        conn = pymysql.connect(
            host=cfg.mysql.host,
            port=cfg.mysql.port,
            user=cfg.mysql.user,
            password=cfg.mysql.password,
            database=cfg.mysql.database,
            charset=cfg.mysql.charset,
            cursorclass=pymysql.cursors.DictCursor,
        )
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()  # 归还给连接池（非真正关闭）


# ─── 审计日志表 ────────────────────────────────────────────────────────────────
# 与应用代码解耦：表结构在这里定义，应用启动时幂等建表（兼容未执行 init.sql 的存量库）。
_AUDIT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT DEFAULT NULL,
    username VARCHAR(50) DEFAULT '',
    action VARCHAR(50) NOT NULL,
    target_type VARCHAR(50) DEFAULT '',
    target_id VARCHAR(64) DEFAULT '',
    detail VARCHAR(1000) DEFAULT '',
    ip VARCHAR(64) DEFAULT '',
    request_id VARCHAR(64) DEFAULT '',
    success TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_action (action),
    KEY idx_user (user_id),
    KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def ensure_audit_table() -> None:
    """确保 audit_log 表存在（幂等），供应用启动时调用。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(_AUDIT_TABLE_DDL)


# ─── 用户相关操作 ──────────────────────────────────────────────────────────────

# ─── 知识库 ────────────────────────────────────────────────────────────────────

def get_kb(kb_id: int) -> Optional[dict]:
    """查询单个知识库（含属主字段，供权限校验）。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, description, domain, is_builtin, owner_id FROM knowledge_bases WHERE id=%s",
            (kb_id,),
        )
        return cur.fetchone()


def ensure_kb_owner_columns() -> None:
    """确保 knowledge_bases.owner_id 列存在（幂等迁移）。

    init.sql 已含该列（FK -> users.id），此处兜底存量库：
    早于属主模型创建的数据库缺列时自动补齐，避免启动失败。
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'knowledge_bases' AND COLUMN_NAME = 'owner_id'"
        )
        if cur.fetchone()["cnt"] > 0:
            return
        logger.warning("knowledge_bases 缺少 owner_id 列，自动补齐（幂等迁移）")
        cur.execute("ALTER TABLE knowledge_bases ADD COLUMN owner_id INT DEFAULT NULL AFTER is_builtin")
        cur.execute("ALTER TABLE knowledge_bases ADD KEY idx_kb_owner (owner_id)")


def get_user_by_username(username: str) -> Optional[dict]:
    """根据用户名获取用户信息。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return cur.fetchone()


def get_user_by_id(user_id: int) -> Optional[dict]:
    """根据用户ID获取用户信息。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        return cur.fetchone()


def create_user(username: str, password: str, email: str = "", role: str = "user") -> int:
    """创建新用户，返回用户ID。"""
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, email, role) VALUES (%s, %s, %s, %s)",
            (username, password_hash, email, role),
        )
        return cur.lastrowid


# 伪造哈希：用户不存在时也执行一次 bcrypt 校验，抹平响应时间差，防止用户枚举
_DUMMY_PASSWORD_HASH = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt()).decode("utf-8")


def verify_password(username: str, password: str) -> Optional[dict]:
    """验证用户名密码，返回用户信息（不含密码哈希）。

    用户不存在时仍执行一次 bcrypt 校验（对伪造哈希），避免通过响应时间差枚举用户名。
    注意：不禁用状态过滤——禁用用户同样要通过本函数验证密码，
    由调用方（登录接口）区分"密码错误"(401) 与"账号被禁用"(403)，
    否则禁用用户会收到误导性的"用户名或密码错误"。
    """
    user = get_user_by_username(username)
    if not user:
        bcrypt.checkpw(password.encode("utf-8"), _DUMMY_PASSWORD_HASH.encode("utf-8"))
        return None
    if bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        user.pop("password_hash", None)
        return user
    return None


def list_users(page: int = 1, page_size: int = 20, search: str = "") -> tuple[list[UserInfo], int]:
    """获取用户列表，返回 (用户列表, 总数)。"""
    with get_db() as conn:
        cur = conn.cursor()
        where = ""
        params: list = []
        if search:
            where = "WHERE username LIKE %s OR email LIKE %s"
            params = [f"%{search}%", f"%{search}%"]

        cur.execute(f"SELECT COUNT(*) as total FROM users {where}", params)
        total = cur.fetchone()["total"]

        offset = (page - 1) * page_size
        cur.execute(
            f"SELECT id, username, role, email, avatar, status, created_at FROM users {where} ORDER BY id DESC LIMIT %s OFFSET %s",
            params + [page_size, offset],
        )
        rows = cur.fetchall()

    users = [
        UserInfo(
            id=r["id"],
            username=r["username"],
            role=r["role"],
            email=r.get("email", ""),
            avatar=r.get("avatar", ""),
            status=r.get("status", "active"),
            created_at=str(r["created_at"]) if r.get("created_at") else None,
        )
        for r in rows
    ]
    return users, total


def update_user(user_id: int, role: Optional[str] = None, status: Optional[str] = None, email: Optional[str] = None) -> bool:
    """更新用户信息。"""
    updates = []
    params = []
    if role is not None:
        updates.append("role=%s")
        params.append(role)
    if status is not None:
        updates.append("status=%s")
        params.append(status)
    if email is not None:
        updates.append("email=%s")
        params.append(email)
    if not updates:
        return False
    params.append(user_id)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=%s", params)
        return cur.rowcount > 0


def delete_user(user_id: int) -> bool:
    """删除用户。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id=%s AND role != 'admin'", (user_id,))
        return cur.rowcount > 0


def get_stats() -> dict:
    """获取系统统计数据。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as total FROM users")
        user_count = cur.fetchone()["total"]
        cur.execute("SELECT COUNT(*) as total FROM users WHERE role='admin'")
        admin_count = cur.fetchone()["total"]
        cur.execute("SELECT COUNT(*) as total FROM knowledge_bases")
        kb_count = cur.fetchone()["total"]
        cur.execute("SELECT COUNT(*) as total FROM documents")
        doc_count = cur.fetchone()["total"]
        cur.execute("SELECT COUNT(*) as total FROM finance_faq")
        faq_count = cur.fetchone()["total"]
    return {
        "user_count": user_count,
        "admin_count": admin_count,
        "kb_count": kb_count,
        "doc_count": doc_count,
        "faq_count": faq_count,
    }


def get_qa_by_category(category: str, limit: int = 20) -> list[BM25Hit]:
    """按领域从 MySQL 查询 FAQ，供 BM25 候选检索使用。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, category, question, answer, source FROM finance_faq WHERE category=%s ORDER BY id DESC LIMIT %s",
            (category, limit),
        )
        rows = cur.fetchall()
        return [BM25Hit(
            question=r["question"],
            answer=r["answer"],
            category=r["category"],
            score=0.0,
            source=r.get("source", "mysql"),
        ) for r in rows]


def get_all_qa_for_bm25() -> list[dict]:
    """返回所有 FAQ 原始记录，用于构建 BM25 索引（含领域过滤）。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, category, question, answer, source, type FROM finance_faq")
        return cur.fetchall()


_SOURCE_MAP_CACHE: dict[str, dict] | None = None
_SOURCE_MAP_CACHE_TS: float = 0.0
_SOURCE_MAP_CACHE_TTL: int = 300  # 缓存有效期（秒），与 BM25 刷新间隔对齐


def get_question_source_map(force: bool = False) -> dict[str, dict]:
    """
    返回 {question: {"source": 来源名, "source_url": 溯源URL}}，供检索层回填来源。
    Milvus 实体未存 source，故用 question 回查 MySQL 补全，便于答案标注出处。
    结果缓存 5 分钟（TTL），force=True 时立即重建；过期后下次调用自动刷新。
    """
    global _SOURCE_MAP_CACHE, _SOURCE_MAP_CACHE_TS
    if _SOURCE_MAP_CACHE is not None and not force and (time.time() - _SOURCE_MAP_CACHE_TS) < _SOURCE_MAP_CACHE_TTL:
        return _SOURCE_MAP_CACHE
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT question, source, source_url FROM finance_faq")
        rows = cur.fetchall()
    m: dict[str, dict] = {}
    for r in rows:
        q = (r.get("question") or "").strip()
        if q:
            m[q] = {"source": r.get("source") or "", "source_url": r.get("source_url") or ""}
    _SOURCE_MAP_CACHE = m
    _SOURCE_MAP_CACHE_TS = time.time()
    return m


def search_qa_exact(question: str, domain: Optional[str] = None) -> dict | None:
    """精确匹配问题（用于高频缓存），返回 {answer, category, ...}。"""
    with get_db() as conn:
        cur = conn.cursor()
        sql = "SELECT question, answer, category, type FROM finance_faq WHERE question=%s"
        params: list = [question]
        if domain:
            sql += " AND category=%s"
            params.append(domain)
        sql += " LIMIT 1"
        cur.execute(sql, params)
        row = cur.fetchone()
        if row:
            return {
                "answer":   row["answer"],
                "category": row["category"],
                "type":     row.get("type", ""),
                "question": row["question"],
            }
        return None
