#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""routers/review.py — 多模态合规审查 API（融合 DocAudit，P1）。

提供：文档上传建任务、SSE/轮询进度事件、任务状态、条款列表、逐条款审查结果、
HITL 人工复核、报告查看。后台流水线由 seeds 服务编排（LangGraph 多 Agent）。

事件通道：Redis List `finrag:compliance:events:{task_id}`，客户端按索引轮询；
任务终态（completed/failed/hitl_pending）由 docaudit_review_task.stage 判定。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from config import get_config
from models import ApiResponse
from routers.auth import get_current_user
from rag_qa.review.compliance_service import init_service, spawn_pipeline
from rag_qa.review.rule_store import RuleRetriever
from rag_qa.review import db as rdb
from services.audit import record_audit

router = APIRouter(prefix="/review", tags=["合规审查"])

# 服务单例 + 规则检索器（首次调用时预热）
_service = None
_retriever = None

# 上传文件白名单：fitz 只能解析 PDF/图片，其余类型提前拒绝（后台报错体验差）
ALLOWED_UPLOAD_EXT = {".pdf", ".png", ".jpg", ".jpeg"}

# 流式读取分片大小（1MB），避免超大文件一次性读入内存
_CHUNK_SIZE = 1024 * 1024


async def _read_upload_limited(file: UploadFile) -> bytes:
    """流式读取上传内容并在累计超限后立即拒绝。

    直接 await file.read() 会把整个文件读进内存（DoS 风险），且超限判断发生在读入之后；
    这里对齐 routers/document.py 的管控约定：边读边累加，超限即 413。
    """
    cfg = get_config()
    max_bytes = cfg.multimodal.upload_max_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件超过 {cfg.multimodal.upload_max_mb}MB 限制",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _load_task(task_id: int, user: dict) -> dict:
    """按 task_id 载入任务并校验属主（防越权读取他人审查任务/报告）。

    管理员可跨用户访问；普通用户仅限自己创建的任务（created_by 记录 username）。
    """
    task = await rdb.afetch_one(
        "SELECT t.*, d.created_by AS _owner "
        "FROM docaudit_review_task t JOIN docaudit_document d ON d.id = t.document_id "
        "WHERE t.id = %s",
        (task_id,),
    )
    if not task:
        raise HTTPException(status_code=404, detail="审查任务不存在")
    owner = task.get("_owner")
    if user.get("role") != "admin" and owner != user.get("username"):
        # 不泄露资源是否存在：统一按 404 返回，避免枚举探测
        raise HTTPException(status_code=404, detail="审查任务不存在")
    task.pop("_owner", None)
    return task


def _get_retriever() -> RuleRetriever:
    global _retriever
    if _retriever is None:
        from services import embedding as emb

        _retriever = RuleRetriever()
        import threading
        def _prewarm():
            try:
                # 先触发 BGE-M3 加载（规则稠密检索依赖），再加载规则
                emb.get_embedding_model()
                _retriever.load_sync()
            except Exception:
                pass
        threading.Thread(target=_prewarm, daemon=True).start()
    return _retriever


def _get_service():
    global _service
    if _service is None:
        _service = init_service(_get_retriever())
    return _service


class ApplyReviewReq(BaseModel):
    action: str  # approved / rejected


@router.post("/upload", response_model=ApiResponse)
async def upload_and_start(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """上传 PDF/图片，创建审查任务并后台启动流水线。"""
    # 先校验扩展名：白名单之外的类型直接拒绝，避免无谓读入（fitz 只支持 PDF/图片）
    doc_name = file.filename or "untitled.pdf"
    suffix = Path(doc_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix or '(无扩展名)'}，仅支持 {', '.join(sorted(ALLOWED_UPLOAD_EXT))}",
        )
    # 流式读取 + 边读边判超限（超限直接 413，不把大文件读进内存）
    file_bytes = await _read_upload_limited(file)
    doc_type = suffix.lstrip(".") or "pdf"
    try:
        svc = _get_service()
        task_id = await svc.create_task(file_bytes, doc_name, doc_type, user.get("username", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    spawn_pipeline(svc, task_id)
    record_audit(
        user.get("user_id"), user.get("username", ""),
        action="review.upload", target_type="review_task", target_id=str(task_id),
        detail=f"{doc_name} ({len(file_bytes)} bytes)", request=request,
    )
    try:
        from services.metrics import record_review_task

        record_review_task("accepted")
    except Exception:  # noqa: BLE001
        pass
    return ApiResponse(success=True, message="审查任务已创建", data={"task_id": task_id})


@router.get("/task/{task_id}/status", response_model=ApiResponse)
async def task_status(task_id: int, user: dict = Depends(get_current_user)):
    """查询任务状态与当前进度（仅任务属主/管理员）。"""
    task = await _load_task(task_id, user)
    progress = task["progress"]
    if isinstance(progress, str):
        try:
            progress = json.loads(progress)
        except json.JSONDecodeError:
            progress = {}
    task["progress"] = progress
    return ApiResponse(success=True, data=task)


@router.get("/task/{task_id}/events", response_model=ApiResponse)
async def task_events(
    task_id: int,
    start: int = Query(0, ge=0),
    _user: dict = Depends(get_current_user),
):
    """按索引轮询流水线事件（SSE 替代：客户端递增 start 拉取增量）。"""
    raw, idx, final = await _get_service().read_events(task_id, start)
    events = []
    for line in raw:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return ApiResponse(
        success=True,
        data={"events": events, "next_start": idx, "final": final},
    )


@router.get("/task/{task_id}/clauses", response_model=ApiResponse)
async def task_clauses(task_id: int, user: dict = Depends(get_current_user)):
    """任务提取的条款列表。仅任务属主/管理员。"""
    task = await _load_task(task_id, user)
    rows = await rdb.afetch_all(
        "SELECT id, clause_no, title, content, page_start, page_end, extract_status, defects "
        "FROM docaudit_clause WHERE document_id = %s ORDER BY id", (task["document_id"],))
    return ApiResponse(success=True, data={"clauses": rows})


@router.get("/task/{task_id}/reviews", response_model=ApiResponse)
async def task_reviews(task_id: int, _user: dict = Depends(get_current_user)):
    """逐条款审查结果与 HITL 状态。"""
    rows = await rdb.afetch_all(
        "SELECT id, clause_id, clause_no, verdict, risk_level, violated_rules, evidence, "
        "suggestion, hitl_status, reviewer, reviewed_at "
        "FROM docaudit_clause_review WHERE task_id = %s ORDER BY id", (task_id,))
    results = []
    for r in rows:
        r["violated_rules"] = (
            json.loads(r["violated_rules"])
            if isinstance(r["violated_rules"], str) else (r["violated_rules"] or []))
        results.append(r)
    return ApiResponse(success=True, data={"reviews": results})


@router.get("/task/{task_id}/report", response_model=ApiResponse)
async def task_report(task_id: int, user: dict = Depends(get_current_user)):
    """合规审查报告（Markdown 原文 + 结构化摘要）。仅任务属主/管理员。"""
    await _load_task(task_id, user)
    row = await rdb.afetch_one(
        "SELECT task_id, summary, md_path FROM docaudit_report WHERE task_id = %s", (task_id,))
    if not row:
        raise HTTPException(status_code=404, detail="报告尚未生成")
    summary = row["summary"]
    if isinstance(summary, str):
        try:
            summary = json.loads(summary)
        except json.JSONDecodeError:
            summary = {}
    md = ""
    if row["md_path"]:
        try:
            md = Path(row["md_path"]).read_text(encoding="utf-8")
        except OSError:
            md = ""
    return ApiResponse(success=True, data={"summary": summary, "markdown": md})


@router.post("/reviews/{review_id}/apply", response_model=ApiResponse)
async def apply_review(
    review_id: int,
    req: ApplyReviewReq,
    request: Request,
    user: dict = Depends(get_current_user),
):
    """HITL 人工复核：approve / reject 待审条款（仅任务属主/管理员，且留审计）。"""
    # 复核属于"写"操作：先按 review_id 反查 task 并校验属主，避免替他人复核
    rel = await rdb.afetch_one(
        "SELECT task_id FROM docaudit_clause_review WHERE id = %s", (review_id,))
    if not rel:
        raise HTTPException(status_code=404, detail="待复核条款不存在")
    await _load_task(rel["task_id"], user)
    try:
        ok = await _get_service().apply_review(review_id, req.action, user.get("username", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="待复核条款不存在")
    record_audit(
        user.get("user_id"), user.get("username", ""),
        action="review.apply", target_type="clause_review", target_id=str(review_id),
        detail=f"action={req.action} task_id={rel['task_id']}", request=request,
    )
    return ApiResponse(success=True, message="复核已提交")


@router.get("/rules", response_model=ApiResponse)
async def list_rules(_user: dict = Depends(get_current_user)):
    """当前加载的监管规则（诊断/核对用）。"""
    retriever = _get_retriever()
    retriever.load_sync() if not retriever._loaded else None  # noqa: SLF001
    return ApiResponse(success=True, data={"rules": retriever._rules})  # noqa: SLF001