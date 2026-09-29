# -*- coding: utf-8 -*-
"""临时运维脚本：盘点知识库与已上传文档（只读）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.mysql import get_db  # noqa: E402


def main():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, description, domain, doc_count, created_at FROM knowledge_bases ORDER BY id")
            kbs = cur.fetchall()
            print("=" * 100)
            print(f"知识库共 {len(kbs)} 个：")
            for kb in kbs:
                print(f"  [id={kb['id']}] {kb['name']} | domain={kb['domain']} | doc_count={kb['doc_count']} | desc={kb.get('description')}")
            print("=" * 100)

            cur.execute(
                "SELECT id, kb_id, filename, status, chunk_count, created_at FROM documents ORDER BY kb_id, id"
            )
            docs = cur.fetchall()
            print(f"文档记录共 {len(docs)} 条：")
            for d in docs:
                print(f"  [doc={d['id']}] kb={d['kb_id']} | {d['filename']} | {d['status']} | chunks={d['chunk_count']} | {d['created_at']}")
            print("=" * 100)

            cur.execute("SELECT category, COUNT(*) AS c FROM finance_faq GROUP BY category ORDER BY c DESC")
            rows = cur.fetchall()
            print("FAQ（finance_faq）分类分布：")
            for r in rows:
                print(f"  {r['category']}: {r['c']}")


if __name__ == "__main__":
    main()
