#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""routers/nl2sql.py — 结构化数据问数 API（融合 ChatBI, P2）。

提供三条链路：
- /ask：自然语言 → SQL Guard → 只读执行（NL2SQL 主链路）；
- /execute：直接粘贴 SQL，仍走 SQLGuard + 只读（不信任外部 SQL）；
- /schema 与 /guard_check：诊断/安全演示用。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from models import ApiResponse
from routers.auth import get_current_user
from rag_qa.nl2sql import sqlguard
from rag_qa.nl2sql.meta_store import MetaStore
from rag_qa.nl2sql.service import Nl2SqlService
from services.audit import record_audit

router = APIRouter(prefix="/nl2sql", tags=["结构化问数"])

_service: Nl2SqlService | None = None


def _get_service() -> Nl2SqlService:
    global _service
    if _service is None:
        _service = Nl2SqlService(MetaStore())
    return _service


class AskReq(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class DirectReq(BaseModel):
    sql: str = Field(min_length=1, max_length=4000)


class GuardCheckReq(BaseModel):
    sql: str = Field(min_length=1, max_length=4000)


@router.post("/ask", response_model=ApiResponse)
async def ask(
    req: AskReq,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """自然语言 → 结构化数据（NL2SQL + SQLGuard 纵深防御 + 只读执行）。"""
    svc = _get_service()
    answer = await svc.ask(req.question, role=user.get("role", "user"))
    # 金融可追溯：记录谁查了什么（含生成 SQL 与涉及表），审计失败不影响业务
    record_audit(
        user.get("user_id"), user.get("username", ""),
        action="nl2sql.ask", target_type="biz_query",
        target_id=",".join(getattr(answer, "tables", None) or [])[:64],
        detail=f"q={req.question[:200]} | sql={(answer.sql or '')[:300]}", request=request,
        success=answer.status == "ok",
    )
    return ApiResponse(success=True, data=answer.model_dump())


@router.post("/execute", response_model=ApiResponse)
async def execute_raw(
    req: DirectReq,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """直接执行一段 SQL（幂等只读演示/高级用途），仍受 SQLGuard 保护。"""
    svc = _get_service()
    answer = await svc.execute_direct(req.sql, role=user.get("role", "user"))
    record_audit(
        user.get("user_id"), user.get("username", ""),
        action="nl2sql.execute", target_type="biz_query",
        detail=f"sql={req.sql[:300]}", request=request,
        success=answer.status == "ok",
    )
    return ApiResponse(success=True, data=answer.model_dump())


@router.get("/schema", response_model=ApiResponse)
async def schema_tables(user: dict = Depends(get_current_user)):
    """当前角色可见的业务表清单（诊断/前端下拉用）。"""
    svc = _get_service()
    await svc.ensure_ready()
    allowed = svc.retriever.allowed_tables_for(user.get("role", "user"))
    payload = []
    for t in svc.meta.table_registry_rows:
        if t["table_name"] in allowed:
            payload.append({
                "table": t["table_name"],
                "display_name": t.get("display_name", ""),
                "description": t.get("description", ""),
            })
    return ApiResponse(success=True, data={"tables": payload})


@router.post("/guard_check", response_model=ApiResponse)
async def guard_check(
    req: GuardCheckReq,
    user: dict = Depends(get_current_user),
):
    """单跑 SQLGuard 校验（防御效果可视化的安全演示接口）。"""
    svc = _get_service()
    await svc.ensure_ready()
    allowed = svc.retriever.allowed_tables_for(user.get("role", "user"))
    result = sqlguard.check(req.sql, allowed)
    return ApiResponse(success=True, data=result.model_dump())