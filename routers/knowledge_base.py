#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
routers/knowledge_base.py — 知识库管理路由
提供知识库列表查询、新增、删除等 CRUD 接口。
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from loguru import logger

from models import KnowledgeBaseInfo, ApiResponse
from db.mysql import get_db, get_kb
from db.chunk import delete_chunks_by_kb
from db.document import get_documents_by_kb, hard_delete_document_record
from routers.auth import get_current_user
from services.audit import record_audit
from services.authz import can_manage_kb
from pathlib import Path

router = APIRouter(prefix="/knowledge_base", tags=["知识库管理"])


# ─── 内置知识库定义 ─────────────────────────────────────────────────────────────
BUILTIN_KB = [
    {"id": 1, "name": "银行业务",           "description": "商业银行、信贷、支付结算等银行业务知识",
     "domain": "banking",           "is_builtin": True},
    {"id": 2, "name": "公司金融",           "description": "企业融资、并购重组、资本结构等公司金融知识",
     "domain": "corporate_finance", "is_builtin": True},
    {"id": 3, "name": "财务会计",           "description": "会计准则、财务报表、审计等财务会计知识",
     "domain": "financial_accounting","is_builtin": True},
]


@router.get("/list", response_model=ApiResponse, summary="获取知识库列表")
def list_knowledge_bases(current_user: dict = Depends(get_current_user)):
    """返回所有知识库（含内置 + 用户自定义）。"""
    with get_db() as conn:
        cur = conn.cursor()
        # doc_count 从 documents 表实时统计，避免与 knowledge_bases.doc_count 字段不同步
        cur.execute(
            "SELECT k.id, k.name, k.description, k.domain, k.is_builtin, k.owner_id, k.created_at, "
            "(SELECT COUNT(*) FROM documents d WHERE d.kb_id = k.id AND d.status = 'indexed') AS real_doc_count "
            "FROM knowledge_bases k ORDER BY k.id"
        )
        rows = cur.fetchall()

    custom_kbs = [
        KnowledgeBaseInfo(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            domain=r.get("domain"),
            is_builtin=bool(r["is_builtin"]),
            owner_id=r.get("owner_id"),
            doc_count=r.get("real_doc_count", 0),
            created_at=str(r["created_at"]) if r.get("created_at") else None,
        )
        for r in rows
    ]
    # 合并内置知识库（若无自定义记录）
    existing_ids = {kb.id for kb in custom_kbs}
    for b in BUILTIN_KB:
        if b["id"] not in existing_ids:
            custom_kbs.append(KnowledgeBaseInfo(**b, owner_id=None, doc_count=0, created_at=None))

    logger.info(f"知识库列表查询完成，共 {len(custom_kbs)} 条")
    return ApiResponse(success=True, data=custom_kbs)


@router.post("/", response_model=ApiResponse, summary="创建知识库")
def create_knowledge_base(body: dict, current_user: dict = Depends(get_current_user), request: Request = None):
    """
    创建自定义知识库。
    请求体：{"name": str, "description": str, "domain": str}
    """
    name = body.get("name", "").strip()
    description = body.get("description", "").strip()
    domain = body.get("domain", "")

    if not name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO knowledge_bases (name, description, domain, is_builtin, owner_id, doc_count) "
            "VALUES (%s, %s, %s, 0, %s, 0)",
            (name, description, domain, current_user["user_id"]),
        )
        new_id = cur.lastrowid
    record_audit(current_user["user_id"], current_user["username"], "kb.create",
                 target_type="knowledge_base", target_id=new_id, detail=name, request=request)
    logger.info(f"知识库创建成功: id={new_id}, name={name}")
    return ApiResponse(success=True, message="知识库创建成功", data={"id": new_id, "name": name})


@router.delete("/{kb_id}", response_model=ApiResponse, summary="删除知识库")
def delete_knowledge_base(kb_id: int, current_user: dict = Depends(get_current_user), request: Request = None):
    """删除自定义知识库（属主或管理员；内置知识库仅管理员可删），并清理其 Milvus 向量与磁盘文件。"""
    kb = get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if kb["is_builtin"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="内置知识库仅管理员可删除")
    if not can_manage_kb(current_user, kb):
        raise HTTPException(status_code=403, detail="仅知识库创建者或管理员可删除")

    # 1) 清理该知识库下的 Milvus 向量（避免删除后仍被检索到）
    deleted_vectors = 0
    try:
        deleted_vectors = delete_chunks_by_kb(kb_id)
        logger.info(f"知识库 {kb_id} 的向量已清理: {deleted_vectors} 条")
    except Exception as e:
        logger.warning(f"知识库 {kb_id} 向量清理失败（不影响删除）: {e}")

    # 2) 删除该知识库下的文档记录与其磁盘文件（否则成为孤儿记录，占空间）
    docs = get_documents_by_kb(kb_id, limit=10000, include_deleted=True)
    for d in docs:
        fp = d.get("file_path")
        if fp:
            try:
                Path(fp).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"删除知识库文档文件失败: {fp} ({e})")
        hard_delete_document_record(d["id"])
    logger.info(f"知识库 {kb_id} 的文档记录已清理: {len(docs)} 条")

    # 3) 删除知识库记录
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM knowledge_bases WHERE id=%s", (kb_id,))
    # 4) 内容已消失，作废旧问答缓存，避免命中已删除知识库的过期答案
    try:
        from db.redis import clear_qa_cache
        clear_qa_cache()
    except Exception:
        pass
    logger.info(f"知识库删除成功: id={kb_id}")
    record_audit(current_user["user_id"], current_user["username"], "kb.delete",
                 target_type="knowledge_base", target_id=kb_id,
                 detail=f"deleted_docs={len(docs)}, vectors={deleted_vectors}", request=request)
    return ApiResponse(success=True, message="知识库已删除")


@router.put("/{kb_id}", response_model=ApiResponse, summary="更新知识库信息")
def update_knowledge_base(kb_id: int, body: dict, current_user: dict = Depends(get_current_user)):
    """更新知识库名称、描述与领域（属主或管理员；内置知识库仅管理员）。"""
    name = body.get("name")
    description = body.get("description")
    domain = body.get("domain")

    kb = get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if not can_manage_kb(current_user, kb):
        raise HTTPException(status_code=403, detail="仅知识库创建者或管理员可修改")

    with get_db() as conn:
        cur = conn.cursor()
        updates = []
        params = []
        if name:
            updates.append("name=%s")
            params.append(name)
        if description is not None:
            updates.append("description=%s")
            params.append(description)
        if domain:
            updates.append("domain=%s")
            params.append(domain)
        if updates:
            params.append(kb_id)
            cur.execute(f"UPDATE knowledge_bases SET {', '.join(updates)} WHERE id=%s", params)

    logger.info(f"知识库更新成功: id={kb_id}")
    return ApiResponse(success=True, message="知识库更新成功")
