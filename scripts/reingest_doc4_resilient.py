#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reingest_doc4_resilient.py — 韧性重建文档（默认用于 doc_id=4 注册制 PDF 恢复）。

与 reingest_doc.py 的区别：插入阶段具备「断线自愈」能力。
线上观察到 Milvus standalone 容器在 BGE-M3 模型加载（~1.3GB 内存尖峰）时
可能被 OOM 回收，导致普通 upsert 中途抛错、已写批次丢失。本脚本：
  1) 先删残留（依赖 Milvus，此时通常可用）
  2) 切分 + 一次性把所有块向量化到内存（此阶段不依赖 Milvus）
  3) 分批插入，每批失败即自动 `docker start finrag_milvus` + 重连 + 重试同一批，
     直到写入成功；Milvus 即使中途宕机也能自拉起并完成。

用法：
  .venv/Scripts/python.exe scripts/reingest_doc4_resilient.py \
      --doc-id 4 --kb-id 4 --domain investment_banking \
      --src "D:/FinRag/data/documents/uploads/8906e88267744ff4aff2b9a9c8739c43.pdf" \
      --fname "全面实行股票发行注册制改革投教问答.pdf"
"""
import os
import sys
import shutil
import tempfile
import argparse
import subprocess
import time

sys.path.insert(0, ".")

import db.milvus as milvus_mod
from db.milvus import get_milvus_client, ensure_collection
from config import get_config
from rag_qa.core.document_processor import process_documents
from db.chunk import delete_chunks_by_document, UPSERT_BATCH_SIZE
from db.document import update_document_status, sync_kb_doc_count
from services.embedding import encode_batch_dense_sparse

MILVUS_CONTAINER = "finrag_milvus"
MAX_ATTEMPTS_PER_BATCH = 60
RETRY_WAIT_SEC = 15


def _restart_milvus():
    """尝试拉起 Milvus 容器并等待就绪。"""
    try:
        subprocess.run(["docker", "start", MILVUS_CONTAINER],
                       check=False, capture_output=True, timeout=60)
    except Exception as e:
        print(f"  [warn] docker start 调用失败: {e}")
    # 等待就绪：重置单例后探活
    for _ in range(20):
        try:
            milvus_mod.close_milvus()
            c = get_milvus_client()
            c.query("finrag_faq", "doc_id == 4", output_fields=["id"], limit=1)
            print("  [ok] Milvus 已重新就绪")
            return c
        except Exception:
            time.sleep(RETRY_WAIT_SEC)
    return None


def _build_row(chunk, kb_id, domain, doc_id, dense, sparse):
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
        "metadata": {
            "kb_id":       kb_id,
            "parent_id":   chunk.get("parent_id", ""),
            "domain":      domain,
            "source_file": chunk.get("source_file", ""),
            **({"doc_id": doc_id} if doc_id is not None else {}),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", type=int, required=True)
    ap.add_argument("--kb-id", type=int, required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--fname", required=True)
    args = ap.parse_args()

    cfg = get_config()
    if not os.path.exists(args.src):
        print(f"[abort] 源文件不存在: {args.src}")
        return

    # 1) 删除残留（此时 Milvus 应可用）
    print(f"[1/4] 删除 kb={args.kb_id} 中 doc_id={args.doc_id} 及同名 source_file 残留 ...")
    n1 = delete_chunks_by_document(kb_id=args.kb_id, doc_id=args.doc_id)
    n2 = delete_chunks_by_document(kb_id=args.kb_id, source_file=args.fname)
    print(f"      删除 doc_id={args.doc_id}: {n1} 条；source_file 回退: {n2} 条")

    # 2) 切分
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

    # 3) 构造写入行（与 upsert_chunks 结构一致）
    processed = [{
        "text":           ch.page_content[:65535],
        "parent_content": ch.metadata.get("parent_content", ch.page_content)[:65535],
        "parent_id":      ch.metadata.get("parent_id", ""),
        "id":             ch.metadata.get("id", ""),
        "type":           "",
        "source_file":    args.fname,
    } for ch in chunks]

    # 4) 一次性向量化到内存（不依赖 Milvus）
    print("[3/4] 向量化（BGE-M3）到内存 ...")
    t0 = time.time()
    texts = [p["text"] for p in processed]
    dense_vecs, sparse_vecs = encode_batch_dense_sparse(texts)
    print(f"      向量化完成 {len(processed)} 块，耗时 {time.time()-t0:.1f}s")

    # 5) 分批插入，断线自愈
    print("[4/4] 写入 Milvus（韧性：断线自动重连重试）...")
    client = get_milvus_client()
    ensure_collection(client)
    total = len(processed)
    written = 0
    for start in range(0, total, UPSERT_BATCH_SIZE):
        batch = processed[start:start + UPSERT_BATCH_SIZE]
        data = [_build_row(batch[i], args.kb_id, args.domain, args.doc_id,
                            dense_vecs[start + i], sparse_vecs[start + i])
                for i in range(len(batch))]
        ok = False
        for attempt in range(MAX_ATTEMPTS_PER_BATCH):
            try:
                client.insert(collection_name=cfg.milvus.collection_name, data=data)
                written += len(batch)
                ok = True
                break
            except Exception as e:
                print(f"  [retry] 写入 {written}+{len(batch)} 失败(第{attempt+1}次): {str(e)[:100]}; 尝试拉起 Milvus...")
                client = _restart_milvus()
                if client is None:
                    print("  [fatal] 无法恢复 Milvus 连接，中止")
                    return
        if not ok:
            print(f"  [fatal] 批次 {start} 重试耗尽，中止")
            return
        print(f"      写入 {written}/{total}")

    print("[5] 更新 MySQL 状态与知识库计数 ...")
    update_document_status(args.doc_id, "indexed", chunk_count=written)
    sync_kb_doc_count(args.kb_id)
    print(f"[done] doc_id={args.doc_id} 重建完成，chunk_count={written}")


if __name__ == "__main__":
    main()
