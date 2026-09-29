#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reingest_kb4_doc31.py — 用【修复后的】db/chunk.py 直接重建 kb=4 的「投资者教育文章.pdf」(doc_id=31)。

为什么不用 HTTP 上传接口：
  当前运行中的服务进程是在 db/chunk.py 修复前启动的（12:22:21 启动，修复落地于 12:48:02），
  旧服务仍使用「doc_id 嵌套进 metadata」的写法，删除 filter 永远命中不到、写入也不可按 doc_id 查。
  故此处绕过 HTTP 服务，直接调用修复后的入库管线，确保 doc_id 以扁平动态字段写入，
  与「全面实行股票发行注册制改革投教问答.pdf」(doc_id=4) 的 chunk id 不再互撞。

步骤：
  1) 删除 kb=4 中 doc_id=31 及 source_file=投资者教育文章.pdf 的全部残留块（含旧嵌套写法留下的）
  2) process_documents 解析切分（doc_id=31 → chunk id 前缀 doc_31_）
  3) upsert_chunks 向量化入库（修复后写法：doc_id 扁平字段）
  4) 更新 MySQL documents.chunk_count 与 knowledge_bases.doc_count
"""
import os
import sys
import shutil
import tempfile
import time

sys.path.insert(0, ".")

from config import get_config
from rag_qa.core.document_processor import process_documents
from db.chunk import upsert_chunks, delete_chunks_by_document
from db.document import update_document_status, sync_kb_doc_count

SRC = r"D:\汪欢\Documents\投资补充资料\投资者教育文章.pdf"
KB_ID = 4
DOC_ID = 31
DOMAIN = "investment_banking"
FNAME = "投资者教育文章.pdf"


def main():
    cfg = get_config()
    if not os.path.exists(SRC):
        print(f"[abort] 源文件不存在: {SRC}")
        return

    # 1) 清理残留（修复后 filter 可精确命中扁平 doc_id；旧嵌套写法留下的块靠 source_file 回退删除）
    print("[1/4] 删除 kb=4 中 doc_id=31 及同名 source_file 的残留向量 ...")
    n1 = delete_chunks_by_document(kb_id=KB_ID, doc_id=DOC_ID)
    n2 = delete_chunks_by_document(kb_id=KB_ID, source_file=FNAME)
    print(f"      删除 doc_id={DOC_ID}: {n1} 条；source_file 回退删除: {n2} 条")

    # 2) 解析 + 双层切分
    print("[2/4] 解析并切分文档 ...")
    tmp_dir = tempfile.mkdtemp(prefix="reingest_")
    try:
        shutil.copy2(SRC, os.path.join(tmp_dir, FNAME))
        chunks = process_documents(
            tmp_dir,
            parent_chunk_size=cfg.retrieval.parent_chunk_size,
            child_chunk_size=cfg.retrieval.child_chunk_size,
            chunk_overlap=cfg.retrieval.chunk_overlap,
            doc_id=DOC_ID,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"      切分出 {len(chunks)} 个子块")

    # 3) 向量化入库（修复后 db/chunk.py：doc_id 扁平字段）
    print("[3/4] 向量化并写入 Milvus（doc_id 扁平字段）...")
    processed = []
    for ch in chunks:
        processed.append({
            "text":           ch.page_content[:65535],
            "parent_content": ch.metadata.get("parent_content", ch.page_content)[:65535],
            "parent_id":      ch.metadata.get("parent_id", ""),
            "id":             ch.metadata.get("id", ""),
            "type":           "",
            "source_file":    FNAME,
        })
    t0 = time.time()
    write_count = upsert_chunks(
        kb_id=KB_ID,
        domain=DOMAIN,
        chunks=processed,
        doc_id=DOC_ID,
    )
    print(f"      写入 {write_count} 条，耗时 {time.time()-t0:.1f}s")

    # 4) 落库状态
    print("[4/4] 更新 MySQL 文档状态与知识库计数 ...")
    update_document_status(DOC_ID, "indexed", chunk_count=write_count)
    sync_kb_doc_count(KB_ID)
    print(f"[done] doc_id={DOC_ID} 重建完成，chunk_count={write_count}")


if __name__ == "__main__":
    main()
