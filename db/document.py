#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
db/document.py — 文档入库操作（MySQL 记录 + Milvus 向量索引）
"""

from typing import Optional
from loguru import logger
from config import get_config
from db.mysql import get_db


def add_document_record(
    kb_id: int,
    filename: str,
    file_path: str,
    status: str = "uploaded",
    chunk_count: int = 0,
) -> int:
    """向 MySQL documents 表插入一条文档记录，返回新记录 id。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO documents (kb_id, filename, file_path, status, chunk_count, created_at)
               VALUES (%s, %s, %s, %s, %s, NOW())""",
            (kb_id, filename, file_path, status, chunk_count),
        )
        doc_id = cur.lastrowid
    logger.info(f"文档记录已写入 MySQL: id={doc_id}, kb_id={kb_id}, filename={filename}")
    return doc_id


def update_document_status(doc_id: int, status: str, chunk_count: int = 0) -> None:
    """更新文档状态和分块数。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE documents SET status=%s, chunk_count=%s, updated_at=NOW()
               WHERE id=%s""",
            (status, chunk_count, doc_id),
        )
    logger.info(f"文档状态已更新: id={doc_id}, status={status}, chunks={chunk_count}")


def get_documents_by_kb(kb_id: int, limit: int = 20, offset: int = 0, include_deleted: bool = False) -> list[dict]:
    """查询某知识库下的文档列表（分页，默认不含已删除记录）。"""
    where = "kb_id=%s" if include_deleted else "kb_id=%s AND status != 'deleted'"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""SELECT id, kb_id, filename, file_path, status, chunk_count, created_at
               FROM documents WHERE {where} ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            (kb_id, limit, offset),
        )
        return cur.fetchall()


def get_document_count_by_kb(kb_id: int) -> int:
    """查询某知识库下的文档总数。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM documents WHERE kb_id=%s", (kb_id,))
        row = cur.fetchone()
        return row["cnt"] if row else 0


def get_document(doc_id: int) -> dict | None:
    """查询单条文档记录。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE id=%s", (doc_id,))
        return cur.fetchone()


def delete_document_record(doc_id: int) -> bool:
    """删除文档记录（同时软删除，status 置为 'deleted'）。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE documents SET status='deleted', updated_at=NOW() WHERE id=%s",
            (doc_id,),
        )
        return cur.rowcount > 0


def hard_delete_document_record(doc_id: int) -> bool:
    """物理删除文档记录（用户显式删除文档时使用，避免列表残留）。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
        return cur.rowcount > 0


def get_same_file_docs(kb_id: int, filename: str, exclude_doc_id: Optional[int] = None) -> list[dict]:
    """查询同知识库下同名（且未删除）的其他文档记录。旧数据 Milvus 向量按 source_file 匹配时使用。"""
    with get_db() as conn:
        cur = conn.cursor()
        sql = "SELECT id, status FROM documents WHERE kb_id=%s AND filename=%s AND status != 'deleted'"
        params: list = [kb_id, filename]
        if exclude_doc_id is not None:
            sql += " AND id != %s"
            params.append(exclude_doc_id)
        cur.execute(sql, params)
        return cur.fetchall()


def sync_kb_doc_count(kb_id: int) -> int:
    """
    根据 documents 表实时同步 knowledge_bases.doc_count。
    返回同步后的文档数。
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE knowledge_bases SET doc_count = ("
            "  SELECT COUNT(*) FROM documents d WHERE d.kb_id = %s AND d.status = 'indexed'"
            ") WHERE id = %s",
            (kb_id, kb_id),
        )
        cur.execute("SELECT doc_count FROM knowledge_bases WHERE id = %s", (kb_id,))
        row = cur.fetchone()
        count = row["doc_count"] if row else 0
    logger.info(f"知识库文档计数已同步: kb_id={kb_id}, doc_count={count}")
    return count


def reconcile_stale_documents() -> int:
    """启动时清理孤儿状态：将所有 status='parsing' 的文档标记为 'failed'。

    服务重启后，后台处理线程已丢失，parsing 状态的文档永远不会完成。
    此函数在应用启动时调用，将它们标记为 failed，让用户知道需要重新上传。
    返回受影响的文档数。
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, kb_id, filename FROM documents WHERE status = 'parsing'"
        )
        stale_docs = cur.fetchall()
        if not stale_docs:
            return 0
        doc_ids = [d["id"] for d in stale_docs]
        # 批量更新为 failed
        placeholders = ", ".join(["%s"] * len(doc_ids))
        cur.execute(
            f"UPDATE documents SET status='failed', updated_at=NOW() WHERE id IN ({placeholders})",
            doc_ids,
        )
    for d in stale_docs:
        logger.warning(
            f"孤儿文档标记为 failed: id={d['id']}, kb_id={d['kb_id']}, filename={d['filename']}"
        )
    logger.info(f"启动清理完成: {len(stale_docs)} 个 parsing 状态文档已标记为 failed")
    return len(stale_docs)
