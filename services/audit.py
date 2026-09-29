#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/audit.py — 审计日志服务

金融合规场景需要可追溯的操作留痕：谁登录、上传/删除了什么文档、
创建/删除了哪个知识库、管理员改了什么，都应记录到独立的 audit_log 表。
- 表在应用启动时自动创建（幂等，见 db.mysql.ensure_audit_table）
- 写入失败只告警不抛出（审计不得阻断业务，fail-open）
"""

from typing import Optional

from loguru import logger

from db.mysql import get_db


def record_audit(
    user_id: Optional[int],
    username: str,
    action: str,
    target_type: str = "",
    target_id: str = "",
    detail: str = "",
    request: Optional[object] = None,
    success: bool = True,
) -> None:
    """写入一条审计日志。

    request 可选：传入 FastAPI Request 时自动提取来源 IP 与 request_id。
    任何异常都只记录告警，不向调用方抛出。
    """
    ip = ""
    request_id = ""
    if request is not None:
        try:
            client = getattr(request, "client", None)
            ip = client.host if client else ""
            request_id = (
                getattr(request.state, "request_id", None)
                or request.headers.get("X-Request-ID", "")
            )
        except Exception:  # noqa: BLE001
            ip, request_id = "", ""

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO audit_log
                   (user_id, username, action, target_type, target_id, detail, ip, request_id, success)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    user_id,
                    (username or "")[:50],
                    action[:50],
                    target_type[:50],
                    str(target_id)[:64],
                    (detail or "")[:1000],
                    ip[:64],
                    request_id[:64],
                    1 if success else 0,
                ),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"审计日志写入失败（不影响业务）: action={action}, err={e}")
