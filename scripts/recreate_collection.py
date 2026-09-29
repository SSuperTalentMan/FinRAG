#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/recreate_collection.py — 人工确认后重建 Milvus collection

背景：db/milvus.py::ensure_collection 检测到 schema 不匹配时不再静默 drop
（避免清空全部已入库向量），而是抛错要求人工介入。本脚本即人工介入的执行入口。

危险操作：将删除 collection 下的全部向量数据（FAQ + 文档块）。
执行后需要重新导入数据：
    1. python init_data.py                 # FAQ 数据
    2. python ingest_blanks.py             # 批量知识库（或 reindex_kb_docs.py 重建已有库）

用法：
    python scripts/recreate_collection.py --yes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from db.milvus import COLLECTION_NAME, ensure_collection, get_milvus_client


def main() -> None:
    parser = argparse.ArgumentParser(description="重建 Milvus collection（清空全部数据，需 --yes 确认）")
    parser.add_argument("--yes", action="store_true", help="确认执行危险操作")
    args = parser.parse_args()

    if not args.yes:
        sys.exit(f"拒绝执行：此操作会清空 collection '{COLLECTION_NAME}' 的全部数据。确认无误后加 --yes 重新执行。")

    client = get_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        logger.warning(f"已删除旧 collection: {COLLECTION_NAME}")

    ensure_collection(client)
    logger.info(f"已按最新 schema 重建 collection: {COLLECTION_NAME}。请重新导入数据（init_data.py / ingest_blanks.py 等）。")


if __name__ == "__main__":
    main()
