#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
routers/document.py — 文档上传、解析、入库接口
流程：上传文件 → 保存 → 建档(parsing) → 后台线程解析分块 → 分批向量化写入 Milvus → 状态置 indexed
支持：后台上传（接口立即返回）、解析进度查询、停止解析（回滚已写入向量）、按文档删除
"""

import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from loguru import logger

from config import get_config
from db.mysql import get_db
from db.document import (
    add_document_record,
    get_documents_by_kb,
    get_document_count_by_kb,
    get_document,
    get_same_file_docs,
    hard_delete_document_record,
    update_document_status,
    sync_kb_doc_count,
)
from db.chunk import (
    upsert_chunks,
    search_chunks,
    count_chunks_by_kb,
    delete_chunks_by_kb,
    delete_chunks_by_document,
    DocumentCancelled,
)
from db.redis import clear_qa_cache
from db.mysql import get_kb
from rag_qa.core.document_processor import process_documents
from models import ApiResponse
from routers.auth import get_current_user
from services.audit import record_audit
from services.authz import can_manage_kb

router = APIRouter(prefix="/document", tags=["文档管理"])

cfg = get_config()

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
UPLOAD_DIR = Path(cfg.app.documents_path) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# domain 合法格式（与 db/milvus.py 的过滤表达式白名单一致）：
# 非法字符会导致 chunk 的 category 永远匹配不上检索 filter，块成为"不可达数据"
_DOMAIN_RE = re.compile(r"^[a-zA-Z0-9_]{1,50}$")


def _resolve_upload_domain(provided: Optional[str], kb_domain: Optional[str]) -> str:
    """解析上传文档的领域标签（纯函数，便于单测）。

    优先级：显式传入 > 知识库 domain > "general"。
    检索按 chunk 的 category == intent.domain 过滤，若上传不传 domain，
    chunk 会落到 general，非 general 意图的查询永远命中不了（前端上传即不传 domain）。
    显式传入的 domain 不合法时抛 400；回退值不合法时降级为 general。
    """
    if provided:
        if not _DOMAIN_RE.match(provided):
            raise HTTPException(
                status_code=400,
                detail=f"domain 仅允许字母/数字/下划线（≤50 位）: {provided!r}",
            )
        return provided
    fallback = (kb_domain or "").strip()
    return fallback if _DOMAIN_RE.match(fallback) else "general"

# 单文件上传上限（50MB），避免超大文件一次性读入内存撑爆进程（DoS 风险）
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# ─── 后台处理任务注册表 ────────────────────────────────────────────────────────
# doc_id -> {"cancel": threading.Event, "kb_id": int, "filename": str,
#            "started_at": float, "done": int, "total": int, "phase": str}
_processing_tasks: dict[int, dict] = {}
_tasks_lock = threading.Lock()

# 串行处理信号量：同一时间只处理一个文档（解析+编码），多个上传排队依次执行，
# 避免并发编码叠加再次吃满 CPU
_process_semaphore = threading.Semaphore(1)


def _update_task(doc_id: int, **fields) -> None:
    with _tasks_lock:
        task = _processing_tasks.get(doc_id)
        if task is not None:
            task.update(fields)


# ─── 请求 / 响应模型 ───────────────────────────────────────────────────────────
from pydantic import BaseModel


class DocumentListResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: list[dict]


# ─── 后台处理 ──────────────────────────────────────────────────────────────────
def _process_document_task(
    doc_id: int,
    kb_id: int,
    domain: Optional[str],
    file_path: Path,
    filename: str,
    cancel_event: threading.Event,
) -> None:
    """在后台线程中完成：排队 → 解析 → 分块 → 分批向量化 → 入库。可随时取消（回滚已写入向量）。"""
    tmp_dir: Optional[Path] = None
    acquired = False
    try:
        # 串行排队：拿到信号量才开始处理；排队期间也可取消（2 秒内响应）
        _update_task(doc_id, phase="queued")
        while not _process_semaphore.acquire(timeout=2):
            if cancel_event.is_set():
                raise DocumentCancelled("排队等待时被取消")
        acquired = True

        _update_task(doc_id, phase="parsing")
        logger.info(f"开始解析文档: {filename} → kb_id={kb_id}, doc_id={doc_id}")

        tmp_dir = Path(tempfile.mkdtemp(dir=UPLOAD_DIR, prefix="upload_"))
        shutil.copy2(file_path, tmp_dir / file_path.name)
        chunks = process_documents(
            str(tmp_dir),
            parent_chunk_size=cfg.retrieval.parent_chunk_size,
            child_chunk_size=cfg.retrieval.child_chunk_size,
            chunk_overlap=cfg.retrieval.chunk_overlap,
            doc_id=doc_id,
        )
        logger.info(f"分块完成: doc_id={doc_id}, {len(chunks)} 个子块")
        if cancel_event.is_set():
            raise DocumentCancelled("解析完成后检测到取消请求")

        _update_task(doc_id, phase="embedding", total=len(chunks), done=0)

        processed_chunks = []
        for chunk in chunks:
            processed_chunks.append({
                "text":           chunk.page_content[:65535],
                "parent_content": chunk.metadata.get("parent_content", chunk.page_content)[:65535],
                "parent_id":      chunk.metadata.get("parent_id", ""),
                "id":             chunk.metadata.get("id", str(uuid.uuid4())),
                "type":           "",
                "source_file":    filename,
            })

        write_count = upsert_chunks(
            kb_id=kb_id,
            domain=domain,
            chunks=processed_chunks,
            doc_id=doc_id,
            should_cancel=cancel_event.is_set,
            progress_cb=lambda done, total: _update_task(doc_id, done=done, total=total),
        )

        update_document_status(doc_id, "indexed", chunk_count=write_count)
        sync_kb_doc_count(kb_id)
        try:
            clear_qa_cache()  # 文档内容变化，作废旧问答缓存，避免返回过期答案
        except Exception:
            pass
        logger.info(f"文档解析完成: doc_id={doc_id}, chunks={write_count}")

    except DocumentCancelled as e:
        update_document_status(doc_id, "cancelled")
        logger.info(f"文档处理已取消: doc_id={doc_id} ({e})")
    except Exception as e:
        logger.error(f"文档解析失败: doc_id={doc_id}, {e}")
        try:
            update_document_status(doc_id, "failed")
        except Exception:
            pass
    finally:
        if acquired:
            _process_semaphore.release()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        with _tasks_lock:
            _processing_tasks.pop(doc_id, None)


# ─── 接口 ──────────────────────────────────────────────────────────────────────
@router.post("/upload", summary="上传文档（后台解析，立即返回）")
def upload_document(
    kb_id: int = Form(..., description="目标知识库 ID"),
    file: UploadFile = File(..., description="文档文件"),
    domain: Optional[str] = Form(None, description="领域标签（可选，默认 general）"),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """
    上传文档后立即返回 doc_id，解析与向量化在后台线程执行。
    通过 GET /document/list 或 GET /document/processing 跟踪进度，POST /document/cancel/{doc_id} 可停止。
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，支持: {ALLOWED_EXTENSIONS}")

    # 检查知识库是否存在 + 属主权限（普通用户仅能向自己创建的知识库上传；
    # 内置知识库属主为系统，仅管理员可写），并取其 domain 作为上传领域回退值
    kb = get_kb(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if not can_manage_kb(current_user, kb):
        raise HTTPException(status_code=403, detail="仅知识库创建者或管理员可上传文档")
    domain = _resolve_upload_domain(domain, kb.get("domain"))

    # 同名文档拦截：同知识库下已有同名的有效文档时拒绝，
    # 防止重复上传白烧 CPU 与存储（如需替换，先删除旧文档再传）
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, status FROM documents WHERE kb_id=%s AND filename=%s "
            "AND status IN ('uploaded','parsing','indexed') LIMIT 1",
            (kb_id, file.filename),
        )
        dup = cur.fetchone()
    if dup:
        raise HTTPException(
            status_code=409,
            detail=f"同名文档已存在（ID: {dup['id']}，状态: {dup['status']}）。如需重新上传，请先在文档列表删除旧文档。",
        )

    # 保存文件：流式写入并累计大小，超出上限立即中止（避免超大文件一次性读入内存撑爆进程）
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / safe_name
    try:
        size = 0
        with file_path.open("wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)  # 1MB 块读取
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail=f"文件过大（上限 {MAX_UPLOAD_SIZE // 1024 // 1024}MB）")
                out.write(chunk)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 插入文档记录
    doc_id = add_document_record(kb_id=kb_id, filename=file.filename, file_path=str(file_path), status="parsing")

    # 注册后台任务并启动
    cancel_event = threading.Event()
    with _tasks_lock:
        _processing_tasks[doc_id] = {
            "cancel": cancel_event,
            "kb_id": kb_id,
            "filename": file.filename,
            "started_at": time.time(),
            "done": 0,
            "total": 0,
            "phase": "queued",
        }
    t = threading.Thread(
        target=_process_document_task,
        args=(doc_id, kb_id, domain, file_path, file.filename, cancel_event),
        daemon=True,
        name=f"doc-process-{doc_id}",
    )
    t.start()

    record_audit(current_user["user_id"], current_user["username"], "document.upload",
                 target_type="document", target_id=doc_id,
                 detail=f"kb_id={kb_id}, filename={file.filename}, domain={domain or 'general'}",
                 request=request)
    return ApiResponse(success=True, message="文档上传成功，后台解析中", data={
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "parsing",
        "domain": domain or "general",
    })


@router.get("/processing", response_model=ApiResponse, summary="查询正在处理的文档任务及进度")
def list_processing(current_user: dict = Depends(get_current_user)):
    """返回所有正在解析/向量化中的文档任务（含进度）。"""
    with _tasks_lock:
        items = [
            {
                "doc_id": doc_id,
                "kb_id": t["kb_id"],
                "filename": t["filename"],
                "phase": t.get("phase", ""),
                "done": t.get("done", 0),
                "total": t.get("total", 0),
                "elapsed_seconds": round(time.time() - t["started_at"], 1),
            }
            for doc_id, t in _processing_tasks.items()
        ]
    return ApiResponse(success=True, data={"items": items})


@router.post("/cancel/{doc_id}", response_model=ApiResponse, summary="停止文档解析（回滚已写入向量）")
def cancel_document(doc_id: int, current_user: dict = Depends(get_current_user), request: Request = None):
    """停止正在后台解析的文档：已写入的向量会被回滚，状态置为 cancelled。"""
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not can_manage_kb(current_user, get_kb(doc["kb_id"])):
        raise HTTPException(status_code=403, detail="仅知识库创建者或管理员可停止解析")

    with _tasks_lock:
        task = _processing_tasks.get(doc_id)
    if task is not None:
        task["cancel"].set()
        logger.info(f"已请求取消文档处理: doc_id={doc_id}")
        record_audit(current_user["user_id"], current_user["username"], "document.cancel",
                     target_type="document", target_id=doc_id, detail="后台解析中取消", request=request)
        return ApiResponse(success=True, message="停止请求已发送，正在回滚已写入数据", data={"doc_id": doc_id})

    # 任务不在注册表（如服务重启后遗留的 parsing 状态）：直接置为 cancelled
    if doc["status"] == "parsing":
        update_document_status(doc_id, "cancelled")
        record_audit(current_user["user_id"], current_user["username"], "document.cancel",
                     target_type="document", target_id=doc_id, detail="遗留 parsing 状态置为 cancelled", request=request)
        return ApiResponse(success=True, message="文档已停止", data={"doc_id": doc_id})
    raise HTTPException(status_code=400, detail=f"文档当前状态为 {doc['status']}，无需停止")


@router.get("/list", response_model=ApiResponse, summary="查询知识库下的文档列表")
def list_documents(
    kb_id: int,
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """分页查询某知识库下的文档列表（不含已删除）。"""
    docs = get_documents_by_kb(kb_id, limit=limit, offset=offset)
    total = get_document_count_by_kb(kb_id)
    return ApiResponse(success=True, data={"total": total, "items": docs})


@router.get("/chunks", response_model=ApiResponse, summary="查询知识库文档块（预览）")
def list_chunks(
    kb_id: int,
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    """查询某知识库下的文档块列表，用于前端预览（支持分页）。"""
    chunks = search_chunks(kb_id=kb_id, top_k=limit, offset=offset)
    total = count_chunks_by_kb(kb_id)
    return ApiResponse(success=True, data={"total": total, "items": chunks})


def _remove_disk_file(file_path: str) -> None:
    """删除文档在磁盘上的原始文件（幂等，忽略不存在/失败）。"""
    if not file_path:
        return
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"删除磁盘文件失败: {file_path} ({e})")


@router.delete("/{doc_id}", response_model=ApiResponse, summary="删除文档（MySQL 记录 + 该文档的 Milvus 向量）")
def delete_document(doc_id: int, current_user: dict = Depends(get_current_user), request: Request = None):
    """
    删除单个文档：仅清除该文档自己的向量（不再波及整个知识库）。
    若文档正在解析，会先停止解析并回滚。
    旧数据无 doc_id 标记时按 source_file 匹配，同知识库同名的其他记录会一并清理并提示。
    """
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not can_manage_kb(current_user, get_kb(doc["kb_id"])):
        raise HTTPException(status_code=403, detail="仅知识库创建者或管理员可删除文档")
    if doc["status"] == "deleted":
        _remove_disk_file(doc.get("file_path", ""))
        hard_delete_document_record(doc_id)
        return ApiResponse(success=True, message="文档已删除")

    kb_id = doc["kb_id"]
    filename = doc["filename"]

    # 1) 若正在解析，先请求停止（后台任务会自行回滚其已写入的向量）
    with _tasks_lock:
        task = _processing_tasks.get(doc_id)
    if task is not None:
        task["cancel"].set()
        logger.info(f"删除文档前先停止解析: doc_id={doc_id}")

    # 2) 删除该文档的向量：优先 doc_id 精确匹配；无命中（旧数据无 doc_id 标记）回退 source_file
    deleted_vectors = 0
    legacy_shared: list[dict] = []
    try:
        deleted_vectors = delete_chunks_by_document(kb_id=kb_id, doc_id=doc_id)
        if deleted_vectors == 0 and filename:
            deleted_vectors = delete_chunks_by_document(kb_id=kb_id, source_file=filename)
            if deleted_vectors > 0:
                # 回退命中：同知识库同名的其他记录向量已被一并清除，需同步删除其记录（否则成空壳）
                legacy_shared = get_same_file_docs(kb_id, filename, exclude_doc_id=doc_id)
    except Exception as e:
        logger.warning(f"Milvus 删除失败（不影响 MySQL 删除）: {e}")

    # 3) 物理删除 MySQL 记录；旧数据同名记录一并删除（它们的向量已被清除，保留会变空壳）
    hard_delete_document_record(doc_id)
    _remove_disk_file(doc.get("file_path", ""))
    for other in legacy_shared:
        other_full = get_document(other["id"])
        hard_delete_document_record(other["id"])
        if other_full:
            _remove_disk_file(other_full.get("file_path", ""))
    affected = [doc_id] + [o["id"] for o in legacy_shared]

    msg = f"文档已删除（清除向量 {deleted_vectors} 条）"
    if legacy_shared:
        msg += f"；旧数据按文件名匹配，同名记录 {', '.join(str(i) for i in affected)} 已一并删除"
    sync_kb_doc_count(kb_id)
    try:
        clear_qa_cache()  # 文档删除后作废相关问答缓存
    except Exception:
        pass
    logger.info(f"文档已删除: doc_ids={affected}, kb_id={kb_id}, vectors={deleted_vectors}")
    record_audit(current_user["user_id"], current_user["username"], "document.delete",
                 target_type="document", target_id=doc_id,
                 detail=f"kb_id={kb_id}, filename={filename}, vectors_deleted={deleted_vectors}",
                 request=request)
    return ApiResponse(success=True, message=msg, data={"doc_ids": affected, "vectors_deleted": deleted_vectors})


def _update_doc_status(doc_id: int, status: str, chunk_count: int = 0) -> None:
    """更新文档状态（辅助函数，向后兼容）。"""
    update_document_status(doc_id, status, chunk_count)
