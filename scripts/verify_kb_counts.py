# -*- coding: utf-8 -*-
"""临时运维脚本：核对每个知识库在 Milvus 中的实际向量数量（只读）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.chunk import count_chunks_by_kb  # noqa: E402
from db.mysql import get_db  # noqa: E402


def main():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, domain, doc_count FROM knowledge_bases ORDER BY id")
            kbs = cur.fetchall()
    print(f"{'kb':>3} | {'name':<24} | {'domain':<22} | {'MySQL doc_count':>16} | {'Milvus 向量数':>13}")
    print("-" * 100)
    for kb in kbs:
        try:
            n = count_chunks_by_kb(kb["id"])
        except Exception as e:  # noqa: BLE001
            n = f"ERR: {e}"
        print(f"{kb['id']:>3} | {kb['name']:<24} | {kb['domain']:<22} | {kb['doc_count']:>16} | {str(n):>13}")


if __name__ == "__main__":
    main()
