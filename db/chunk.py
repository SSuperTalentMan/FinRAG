#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
db/chunk.py — Milvus 文档块向量索引操作
支持：分批插入子块（可取消/带进度）、按 kb_id 查询、按知识库/按文档删除
"""

from typing import Callable, Optional
import time

from loguru import logger
from config import get_config
from db.milvus import get_milvus_client, ensure_collection
from services.embedding import encode_batch_dense_sparse


class DocumentCancelled(Exception):
    """文档处理被用户取消。"""
    pass


# 每批编码+写入的子块数：兼顾取消响应速度与吞吐
UPSERT_BATCH_SIZE = 64


def upsert_chunks(
    kb_id: int,
    domain: Optional[str],
    chunks: list[dict],
    doc_id: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    将一批子块（含 parent_content 和元数据）向量化后分批写入 Milvus。

    参数:
        doc_id: 文档记录 id，写入 metadata 供按文档删除（None 则不写）。
        should_cancel: 每批开始前调用，返回 True 则中止并清理已写入数据。
        progress_cb: 每批完成后回调 (已完成数, 总数)，用于进度展示。

    返回实际写入条数。被取消时抛出 DocumentCancelled。
    """
    if not chunks:
        return 0

    cfg = get_config()
    client = get_milvus_client()
    ensure_collection(client)

    def _build_row(i: int, dense, sparse) -> dict:
        chunk = chunks[i]
        return {
            "dense_vector":  dense,
            "sparse_vector": sparse,
            "text":          chunk.get("text", chunk.get("parent_content", ""))[:65535],
            "parent_text":   chunk.get("parent_content", "")[:65535],
            "category":      domain or "general",
            "type":          chunk.get("type", ""),
            "question":      chunk.get("id", ""),
            "answer":        chunk.get("parent_content", "")[:65535],
            "kb_id":         kb_id,
            "doc_id":        doc_id,
            "source_file":   chunk.get("source_file", ""),
            "parent_id":     chunk.get("parent_id", ""),
            "metadata":      {
                "kb_id":       kb_id,
                "parent_id":   chunk.get("parent_id", ""),
                "domain":      domain,
                "source_file": chunk.get("source_file", ""),
                **({"doc_id": doc_id} if doc_id is not None else {}),
            },
        }

    def _cleanup_inserted(written: int) -> None:
        """取消时清理已写入的向量（循环复核直至清零，规避可见性延迟）。"""
        try:
            if doc_id is None:
                # 无 doc_id 时按本批 question id 精确回滚
                # 转义每个 id 中的引号/反斜杠，防止 filter 表达式注入
                ids = [chunks[i].get("id", "") for i in range(written)]
                quoted = ", ".join(
                    f'"{str(i).replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34))}"'
                    for i in ids if i
                )
                if not quoted:
                    return
                expr = f'kb_id == {int(kb_id)} and question in [{quoted}]'
            else:
                expr = f'kb_id == {int(kb_id)} and doc_id == {int(doc_id)}'

            removed_total = _delete_expr_until_empty(kb_id, expr)
            logger.info(f"已回滚已写入的向量: kb_id={kb_id}, doc_id={doc_id}, 累计删除 {removed_total} 条（写入 {written} 条）")
            if removed_total < written:
                logger.warning(f"回滚不完整: 已写入 {written} 条，仅删除 {removed_total} 条（剩余为尚未可见的行，可稍后按 doc_id 清理）")
        except Exception as e:
            logger.warning(f"回滚已写入向量失败（可能残留少量数据）: {e}")

    total = len(chunks)
    written = 0
    for start in range(0, total, UPSERT_BATCH_SIZE):
        if should_cancel is not None and should_cancel():
            _cleanup_inserted(written)
            raise DocumentCancelled(f"文档处理已取消（已完成 {written}/{total}，已回滚）")

        batch = chunks[start:start + UPSERT_BATCH_SIZE]
        texts = [c.get("text", c.get("parent_content", "")) for c in batch]
        dense_vecs, sparse_vecs = encode_batch_dense_sparse(texts)

        data = [_build_row(start + i, dense_vecs[i], sparse_vecs[i]) for i in range(len(batch))]
        client.insert(collection_name=cfg.milvus.collection_name, data=data)

        written += len(batch)
        logger.debug(f"  写入 Milvus 子块 {written}/{total}")
        if progress_cb is not None:
            try:
                progress_cb(written, total)
            except Exception:
                pass

    logger.info(f"Milvus 写入完成: kb_id={kb_id}, doc_id={doc_id}, 共 {written} 条子块")
    return written


def search_chunks(
    kb_id: int,
    top_k: int = 20,
    offset: int = 0,
) -> list[dict]:
    """
    查询某知识库下的文档块列表（用于前端预览，支持分页）。
    注意：Milvus 行没有 status 字段（那是 MySQL 的概念），不能按 status 过滤，
    否则永远返回空（FAQ 问答行不带 kb_id，天然不会混入）。
    返回 [{id, text, parent_text, category, kb_id, parent_id, metadata}]
    """
    cfg = get_config()
    client = get_milvus_client()
    ensure_collection(client)

    results = client.query(
        collection_name=cfg.milvus.collection_name,
        filter=f'kb_id == {int(kb_id)}',
        limit=top_k,
        offset=offset,
        output_fields=["text", "parent_text", "category", "kb_id", "parent_id", "metadata"],
    )
    return results


def _delete_expr_until_empty(kb_id: int, expr: str, max_rounds: int = 8) -> int:
    """
    按过滤表达式删除向量，并循环复核直至清零。
    Milvus 对刚写入/刚删除的数据存在秒级可见性延迟，单轮删除可能漏掉尚未可见的行，
    因此删除后复核再删，确保删除确定性。返回累计删除条数。
    """
    cfg = get_config()
    client = get_milvus_client()
    removed_total = 0
    for _ in range(max_rounds):
        all_data = client.query(
            collection_name=cfg.milvus.collection_name,
            filter=expr,
            output_fields=["id"],
            limit=16384,
        )
        if not all_data:
            break
        ids = [d["id"] for d in all_data]
        client.delete(collection_name=cfg.milvus.collection_name, ids=ids)
        removed_total += len(ids)
        time.sleep(1.5)  # 等待删除可见性追平
    return removed_total


def _delete_by_filter(kb_id: int, filter_expr: str) -> int:
    """按过滤表达式删除向量：先查出主键 id，再批量删除（单轮，不做复核）。"""
    cfg = get_config()
    client = get_milvus_client()
    all_data = client.query(
        collection_name=cfg.milvus.collection_name,
        filter=filter_expr,
        output_fields=["id"],
        limit=16384,
    )
    if not all_data:
        return 0
    ids = [d["id"] for d in all_data]
    client.delete(collection_name=cfg.milvus.collection_name, ids=ids)
    return len(ids)


def count_chunks_by_kb(kb_id: int) -> int:
    """统计某知识库下的文档块总数（用于预览分页）。"""
    cfg = get_config()
    client = get_milvus_client()
    res = client.query(
        collection_name=cfg.milvus.collection_name,
        filter=f'kb_id == {int(kb_id)}',
        output_fields=["count(*)"],
    )
    try:
        return int(res[0]["count(*)"])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0


def delete_chunks_by_kb(kb_id: int) -> int:
    """删除指定知识库的所有向量数据。"""
    n = _delete_by_filter(kb_id, f'kb_id == {int(kb_id)}')
    logger.info(f"已删除 kb_id={kb_id} 下的 {n} 条向量数据")
    return n


def delete_chunks_by_document(
    kb_id: int,
    doc_id: Optional[int] = None,
    source_file: Optional[str] = None,
) -> int:
    """
    删除指定文档的向量数据（循环复核直至清零，规避可见性延迟）。
    优先按 metadata["doc_id"]（新数据精确匹配）；
    旧数据无 doc_id 时按 metadata["source_file"] 匹配（同 kb 同名文档会一并命中）。
    """
    kb_id_safe = int(kb_id)
    if doc_id is not None:
        n = _delete_expr_until_empty(kb_id_safe, f'kb_id == {kb_id_safe} and doc_id == {int(doc_id)}')
        if n > 0:
            logger.info(f"已删除文档向量: kb_id={kb_id}, doc_id={doc_id}, 共 {n} 条")
            return n
        logger.info(f"doc_id={doc_id} 无精确匹配向量，尝试按 source_file 回退删除")
    if not source_file:
        return 0
    safe = str(source_file).replace("\\", "\\\\").replace('"', '\\"')
    n = _delete_expr_until_empty(kb_id_safe, f'kb_id == {kb_id_safe} and source_file == "{safe}"')
    logger.info(f"已删除文档向量(source_file 回退): kb_id={kb_id}, source_file={source_file}, 共 {n} 条")
    return n
