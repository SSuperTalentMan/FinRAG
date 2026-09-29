#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/db.py — 元数据库 / 业务库连接工具（同步 pymysql + asyncio 包装）。

FinRag 主链路用同步 pymysql；P2 复用同一套凭证并加只读账号约束：
- meta（chatbi_meta）：管理员凭证，仅做元数据读取；
- biz（biz_demo）：只读账号 chatbi_ro，仅授 SELECT；执行器还会再设会话级 READ ONLY。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import pymysql
from pymysql.cursors import DictCursor

from config import get_config

logger = logging.getLogger(__name__)


def _require_ro_password() -> str:
    """业务库只读账号密码：缺失即 fail-fast，绝不回落到硬编码默认口令。

    历史上这里内置过一个弱口令默认值，等于把弱口令写进代码——一旦有人忘记改配置，
    系统会静默地用所有人都能猜到的密码连生产库。这里改成显式报错，
    运维第一时间就能看到配置缺口。
    """
    pwd = (get_config().nl2sql.ro_password or "").strip()
    if not pwd:
        raise RuntimeError(
            "未配置业务库只读账号密码：[nl2sql] ro_password 为空。"
            "请在 config.ini 的 [nl2sql] 段填写，或设置环境变量 NL2SQL_RO_PASSWORD。"
            "（不设默认值是刻意的：避免以弱口令静默连库）"
        )
    return pwd


def _base_kwargs() -> dict:
    c = get_config()
    m = c.mysql
    return {"host": m.host, "port": m.port, "charset": "utf8mb4", "cursorclass": DictCursor}


def _pooled(creator_kwargs: dict, tag: str):
    """按凭据建一个进程级只读连接池（懒初始化，单例）。

    融合前这里每次查询都新建短连接：问数一个请求要跑元数据 + 业务库多次查询，
    并发上来会瞬间把 MySQL 连接数打满（且 TLS 握手开销全在用户等待时间里）。
    这里对齐主库 db/mysql.py 的 PooledDB 约定；dbutils 缺失时降级为短连接。
    """
    try:
        from dbutils.pooled_db import PooledDB
    except ImportError:  # 未安装依赖：退回原行为，不影响功能
        return None
    try:
        return PooledDB(
            creator=pymysql,
            maxconnections=8,   # 问数为旁支能力，池子比主库小，避免抢占主链路连接
            mincached=1,
            maxcached=4,
            maxshared=0,
            blocking=True,      # 池满时等待而非报错
            ping=1,             # 取连接时探活，避免拿到被 MySQL 断掉的死连接
            **creator_kwargs,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"NL2SQL 连接池初始化失败({tag})，降级为短连接: {e}")
        return None


_meta_pool = None
_biz_pool = None


def _meta_creator_kwargs() -> dict:
    c = get_config()
    m = c.mysql
    kw = _base_kwargs()
    kw.update(user=m.user, password=m.password, database=c.nl2sql.meta_database, autocommit=True)
    return kw


def _biz_creator_kwargs() -> dict:
    c = get_config()
    n = c.nl2sql
    kw = _base_kwargs()
    kw.update(user=n.ro_user, password=_require_ro_password(),
              database=n.biz_database, autocommit=True)
    return kw


def _conn_factory(kind: str):
    """返回 (取连接可调用对象, 归还/关闭处理函数)。"""
    global _meta_pool, _biz_pool
    if kind == "meta":
        if _meta_pool is None:
            _meta_pool = _pooled(_meta_creator_kwargs(), "meta")
        pool = _meta_pool
        return (pool.connection if pool is not None else _meta_conn)
    if _biz_pool is None:
        _biz_pool = _pooled(_biz_creator_kwargs(), "biz")
    pool = _biz_pool
    return pool.connection if pool is not None else _biz_conn


def _close_nl2sql_pools() -> None:
    """关闭问数连接池（供优雅停机调用；未初始化时静默跳过）。"""
    global _meta_pool, _biz_pool
    for name in ("_meta_pool", "_biz_pool"):
        pool = globals().get(name)
        if pool is not None:
            try:
                pool.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"关闭 NL2SQL 连接池异常({name}): {e}")
            globals()[name] = None


def _meta_conn():
    c = get_config()
    m = c.mysql
    kw = _base_kwargs()
    kw.update(user=m.user, password=m.password, database=c.nl2sql.meta_database, autocommit=True)
    return pymysql.connect(**kw)


def _biz_conn():
    c = get_config()
    n = c.nl2sql
    kw = _base_kwargs()
    kw.update(user=n.ro_user, password=_require_ro_password(),
              database=n.biz_database, autocommit=True)
    return pymysql.connect(**kw)


def biz_conn():
    """给执行器用的公开业务库只读连接（分配连接实例，调用方负责 close 归还）。"""
    return _conn_factory("biz")()


def _run(kind: str, fn) -> object:
    """取连接 → 执行 → 归还（池化连接的 close 即归还，短连接即关闭）。"""
    conn = _conn_factory(kind)()
    try:
        return fn(conn)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def fetch_all(sql: str, params: tuple | list | None = None, *, meta: bool = True) -> list[dict[str, Any]]:
    def _q(conn):
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return [dict(r) for r in cur.fetchall()]
    return _run("meta" if meta else "biz", _q)  # type: ignore[return-value]


def fetch_one(sql: str, params: tuple | list | None = None, *, meta: bool = True):
    rows = fetch_all(sql, params, meta=meta)
    return rows[0] if rows else None


def execute(sql: str, params: tuple | list | None = None, *, meta: bool = True) -> int:
    def _x(conn):
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.rowcount
    return _run("meta" if meta else "biz", _x)  # type: ignore[return-value]


# ─── asyncio 包装（主链路事件循环内避免阻塞）───────────────────────────────────
async def afetch_all(sql: str, params: tuple | list | None = None, *, meta: bool = True):
    return await asyncio.to_thread(fetch_all, sql, params, meta=meta)


async def aexecute(sql: str, params: tuple | list | None = None, *, meta: bool = True) -> int:
    return await asyncio.to_thread(execute, sql, params, meta=meta)


def with_biz_conn(runner: Callable[[Any], Any]):
    """在业务库只读连接上执行一段同步逻辑（executor 用它做会话级加固）。"""
    return _run("biz", runner)