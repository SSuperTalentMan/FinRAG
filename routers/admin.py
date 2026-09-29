#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
routers/admin.py — 管理员路由
提供用户管理、系统统计等管理员专属接口。
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from loguru import logger

from models import ApiResponse, UserUpdateRequest
from db.mysql import list_users, update_user, delete_user, get_stats, get_user_by_id
from db.redis import get_redis, SESSION_KEY_PREFIX, SESSION_INDEX_PREFIX
# 别名导入：路由函数 get_user_sessions 与存储层助手重名，避免模块级命名覆盖
from db.redis import get_user_sessions as _fetch_user_sessions
from routers.auth import get_admin_user
from services.audit import record_audit
import json

router = APIRouter(prefix="/admin", tags=["管理员"])


# ─── 用户管理 ──────────────────────────────────────────────────────────────────

@router.get("/users", response_model=ApiResponse, summary="获取用户列表")
def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query(""),
    current_user: dict = Depends(get_admin_user),
):
    """分页获取用户列表（管理员专属）。"""
    users, total = list_users(page=page, page_size=page_size, search=search)
    return ApiResponse(success=True, data={
        "items": users,
        "total": total,
        "page": page,
        "page_size": page_size,
    })


@router.put("/users/{user_id}", response_model=ApiResponse, summary="更新用户信息")
def update_user_info(
    user_id: int,
    req: UserUpdateRequest,
    current_user: dict = Depends(get_admin_user),
    request: Request = None,
):
    """更新用户角色、状态、邮箱等信息（管理员专属）。"""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 不允许修改超级管理员的角色
    if user["username"] == "admin" and req.role and req.role != "admin":
        raise HTTPException(status_code=403, detail="不能修改超级管理员角色")

    success = update_user(
        user_id,
        role=req.role,
        status=req.status,
        email=req.email,
    )
    if success:
        logger.info(f"管理员 {current_user['username']} 更新用户 {user_id} 信息")
        record_audit(current_user["user_id"], current_user["username"], "admin.user_update",
                     target_type="user", target_id=user_id,
                     detail=f"role={req.role}, status={req.status}, email={req.email}",
                     request=request)
        return ApiResponse(success=True, message="更新成功")
    record_audit(current_user["user_id"], current_user["username"], "admin.user_update",
                 target_type="user", target_id=user_id, detail="更新失败", request=request, success=False)
    return ApiResponse(success=False, message="更新失败")


@router.delete("/users/{user_id}", response_model=ApiResponse, summary="删除用户")
def delete_user_account(
    user_id: int,
    current_user: dict = Depends(get_admin_user),
    request: Request = None,
):
    """删除用户（管理员专属，不能删除管理员），同时清理该用户的 Redis 会话。"""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user["role"] == "admin":
        raise HTTPException(status_code=403, detail="不能删除管理员账号")

    success = delete_user(user_id)
    if success:
        # 清理该用户在 Redis 中的所有会话（走会话索引；索引缺失时 helper 内部回退 scan）
        r = get_redis()
        deleted_count = 0
        for sid, _data in _fetch_user_sessions(user_id):
            r.delete(f"{SESSION_KEY_PREFIX}{sid}")
            deleted_count += 1
        r.delete(f"{SESSION_INDEX_PREFIX}{user_id}")
        logger.info(f"管理员 {current_user['username']} 删除用户 {user_id}，清理 {deleted_count} 条会话")
        record_audit(current_user["user_id"], current_user["username"], "admin.user_delete",
                     target_type="user", target_id=user_id,
                     detail=f"username={user['username']}, cleaned_sessions={deleted_count}",
                     request=request)
        return ApiResponse(success=True, message=f"删除成功，已清理 {deleted_count} 条会话")
    record_audit(current_user["user_id"], current_user["username"], "admin.user_delete",
                 target_type="user", target_id=user_id, detail="删除失败", request=request, success=False)
    return ApiResponse(success=False, message="删除失败")


# ─── 系统统计 ──────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=ApiResponse, summary="获取系统统计")
def get_system_stats(current_user: dict = Depends(get_admin_user)):
    """获取系统统计数据（管理员专属）。"""
    stats = get_stats()
    return ApiResponse(success=True, data=stats)


# ─── 用户会话查看 ──────────────────────────────────────────────────────────────

@router.get("/users/{user_id}/sessions", response_model=ApiResponse, summary="获取指定用户的会话列表")
def get_user_sessions(
    user_id: int,
    current_user: dict = Depends(get_admin_user),
):
    """管理员查看指定用户的所有会话列表（走会话索引，避免全库扫描）。"""
    sessions = []
    for sid, data in _fetch_user_sessions(user_id):
        msgs = data.get("messages", [])
        last_ts = max((m["ts"] for m in msgs), default=0) if msgs else data.get("created_at", 0)
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


@router.get("/users/{user_id}/sessions/{session_id}/history", response_model=ApiResponse, summary="获取指定用户的会话历史")
def get_user_session_history(
    user_id: int,
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_admin_user),
):
    """管理员查看指定用户的指定会话历史消息。"""
    r = get_redis()
    val = r.get(f"{SESSION_KEY_PREFIX}{session_id}")
    if not val:
        raise HTTPException(status_code=404, detail="会话不存在")
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    data = json.loads(val)
    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="该会话不属于此用户")
    messages = data.get("messages", [])[-limit:]
    return ApiResponse(success=True, data={
        "session_id": session_id,
        "total": len(messages),
        "messages": messages,
    })
