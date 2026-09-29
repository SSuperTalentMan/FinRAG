#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
routers/conversation.py — 多轮对话历史接口
支持：保存对话、查询历史、清空历史
会话按 user_id 隔离，用户只能看到自己的会话，管理员可以查看所有用户的会话。
"""

import json
import uuid
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from loguru import logger

from db.redis import (
    get_redis,
    SESSION_KEY_PREFIX,
    SESSION_TTL,
    append_session_message,
    index_session,
    unindex_session,
    get_user_sessions,
)
from models import ApiResponse
from routers.auth import get_current_user

router = APIRouter(prefix="/conversation", tags=["对话管理"])


# ─── 请求模型 ──────────────────────────────────────────────────────────────────
class SaveMessageRequest(BaseModel):
    session_id: str
    role: str           # "user" | "assistant"
    content: str = Field(..., max_length=16000, description="消息内容（限制长度防止单条消息撑爆 Redis 会话）")
    domain: Optional[str] = None


# ─── Redis 键设计（常量与原子操作统一在 db.redis，见 SESSION_KEY_PREFIX 注释）──


def _session_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{session_id}"


def _load_session(session_id: str) -> dict:
    r = get_redis()
    val = r.get(_session_key(session_id))
    if val:
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return json.loads(val)
    return {"messages": [], "created_at": int(time.time())}


def _save_session(session_id: str, data: dict) -> None:
    r = get_redis()
    r.setex(_session_key(session_id), SESSION_TTL, json.dumps(data, ensure_ascii=False))


def _check_session_owner(session_data: dict, user: dict) -> bool:
    """检查会话是否属于该用户。"""
    session_user_id = session_data.get("user_id")
    return session_user_id == user["user_id"]


# ─── 接口 ──────────────────────────────────────────────────────────────────────
@router.post("/save", summary="保存一条对话消息")
def save_message(req: SaveMessageRequest, current_user: dict = Depends(get_current_user)):
    """保存用户或助手的消息到会话历史。"""
    if req.role not in ("user", "assistant"):
        raise HTTPException(status_code=400, detail="role 必须是 user 或 assistant")

    session = _load_session(req.session_id)
    # 验证会话归属
    if not _check_session_owner(session, current_user):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    # 原子追加（Lua），并同步刷新会话索引
    append_session_message(
        req.session_id,
        {
            "role":    req.role,
            "content": req.content,
            "domain":  req.domain,
            "ts":      int(time.time()),
        },
        user_id=current_user["user_id"],
    )
    logger.debug(f"消息已保存: session={req.session_id[:8]}..., role={req.role}, user={current_user['username']}")
    return ApiResponse(success=True, message="消息已保存")


@router.get("/history", summary="查询对话历史")
def get_history(
    session_id: str = Query(..., description="会话ID"),
    limit: int = Query(50, ge=1, le=100, description="返回条数"),
    current_user: dict = Depends(get_current_user),
):
    """查询某会话的历史消息（按时间正序返回最近 N 条）。"""
    session = _load_session(session_id)
    # 验证会话归属（管理员可查看任意会话）
    if not _check_session_owner(session, current_user) and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="无权查看此会话")

    messages = session.get("messages", [])[-limit:]  # 取最近 N 条，正序
    return ApiResponse(success=True, data={
        "session_id": session_id,
        "total": len(messages),
        "messages": messages,
    })


@router.delete("/clear", summary="清空会话历史")
def clear_history(
    session_id: str = Query(..., description="会话ID"),
    current_user: dict = Depends(get_current_user),
):
    """清空某会话的对话历史。"""
    session = _load_session(session_id)
    if not _check_session_owner(session, current_user):
        raise HTTPException(status_code=403, detail="无权操作此会话")

    r = get_redis()
    r.delete(_session_key(session_id))
    unindex_session(current_user["user_id"], session_id)
    logger.info(f"会话历史已清空: session={session_id[:8]}..., user={current_user['username']}")
    return ApiResponse(success=True, message="会话历史已清空")


@router.get("/list", summary="获取当前用户的会话列表")
def list_sessions(current_user: dict = Depends(get_current_user)):
    """返回当前用户的所有活跃会话（按最后消息时间排序）。

    走 finrag:user_sessions:{uid} ZSET 索引 + pipeline 批量取值（避免 N+1 与全库扫描）；
    索引缺失时自动回退 scan 并重建。
    """
    uid = current_user["user_id"]
    sessions = []
    for sid, data in get_user_sessions(uid):
        msgs = data.get("messages", [])
        last_ts = max((m["ts"] for m in msgs), default=0) if msgs else data.get("created_at", 0)
        # 提取第一条用户消息作为会话标题
        first_user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        title = first_user_msg[:30] if first_user_msg else "新会话"
        sessions.append({
            "session_id": sid,
            "title": title,
            "message_count": len(msgs),
            "last_active": last_ts,
        })
    sessions.sort(key=lambda x: x["last_active"], reverse=True)
    return ApiResponse(success=True, data=sessions)


@router.post("/new", summary="创建新会话")
def create_session(current_user: dict = Depends(get_current_user)):
    """创建新会话，绑定当前用户并登记会话索引，返回 session_id。"""
    session_id = uuid.uuid4().hex
    now = int(time.time())
    _save_session(session_id, {
        "user_id": current_user["user_id"],
        "messages": [],
        "created_at": now,
    })
    index_session(current_user["user_id"], session_id, now)
    logger.info(f"新会话已创建: {session_id[:8]}..., user={current_user['username']}")
    return ApiResponse(success=True, data={"session_id": session_id})
