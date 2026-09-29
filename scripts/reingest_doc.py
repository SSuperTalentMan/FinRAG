#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reingest_doc.py — 用【修复后的】入库管线直接重建某个文档（绕过旧 HTTP 服务）。

适用场景：历史上传用旧 `doc_0` 前缀导致 chunk id 跨文档互撞、被清理误删的文档；
或任何需要以修复后代码（扁平 doc_id + 全局父块序号）重建的文档。

用法：
    .venv/Scripts/python.exe scripts/reingest_doc.py --doc-id 10 --kb-id 1 \
        --domain banking --src "D:/汪欢/Documents/投资补充资料/banking-常见问题解答.txt" \
        --fname "banking-常见问题解答.txt"
"""
import os
import sys
import shutil
import tempfile
import argparse
import time

sys.path.insert(0, ".")

from config import get_config
from rag_qa.core.document_processor import process_documents
from db.chunk import upsert_chunks, delete_chunks_by_document
from db.document import update_document_status, sync_kb_doc_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--kb-id", type=int, required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--src", required=True, help="源文件路径")
    ap.add_argument("--fname", required=True, help="写入 Milvus 的 source_file 名")
    args = ap.parse_args()

    cfg = get_config()
    if not os.path.exists(args.src):
        print(f"[abort] 源文件不存在: {args.src}")
        return

    print(f"[1/4] 删除 kb={args.kb_id} 中 doc_id={args.doc_id} 及同名 source_file 的残留向量 ...")
    n1 = delete_chunks_by_document(kb_id=args.kb_id, doc_id=args.doc_id)
    n2 = delete_chunks_by_document(kb_id=args.kb_id, source_file=args.fname)
    print(f"      删除 doc_id={args.doc_id}: {n1} 条；source_file 回退: {n2} 条")

    print("[2/4] 解析并切分文档 ...")
    tmp_dir = tempfile.mkdtemp(prefix="reingest_")
    try:
        shutil.copy2(args.src, os.path.join(tmp_dir, args.fname))
        chunks = process_documents(
            tmp_dir,
            parent_chunk_size=cfg.retrieval.parent_chunk_size,
            child_chunk_size=cfg.retrieval.child_chunk_size,
            chunk_overlap=cfg.retrieval.chunk_overlap,
            doc_id=args.doc_id,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"      切分出 {len(chunks)} 个子块")

    print("[3/4] 向量化并写入 Milvus（修复后：doc_id 扁平 + 全局父块序号）...")
    processed = [{
        "text":           ch.page_content[:65535],
        "parent_content": ch.metadata.get("parent_content", ch.page_content)[:65535],
        "parent_id":      ch.metadata.get("parent_id", ""),
        "id":             ch.metadata.get("id", ""),
        "type":           "",
        "source_file":    args.fname,
    } for ch in chunks]
    t0 = time.time()
    write_count = upsert_chunks(
        kb_id=args.kb_id, domain=args.domain,
        chunks=processed, doc_id=args.doc_id,
    )
    print(f"      写入 {write_count} 条，耗时 {time.time()-t0:.1f}s")

    print("[4/4] 更新 MySQL 状态与知识库计数 ...")
    update_document_status(args.doc_id, "indexed", chunk_count=write_count)
    sync_kb_doc_count(args.kb_id)
    print(f"[done] doc_id={args.doc_id} 重建完成，chunk_count={write_count}")


if __name__ == "__main__":
    main()
