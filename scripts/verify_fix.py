import sys
sys.path.insert(0, ".")
from db.milvus import get_milvus_client
from collections import defaultdict

client = get_milvus_client()
COLL = "finrag_faq"

# 全部带 doc_id 的文档块（doc_id>0 是合法 int 过滤，避免 None 比较报错）
rows = client.query(COLL, 'doc_id > 0', output_fields=["kb_id", "doc_id"], limit=16384)
print(f"[total] 带 doc_id 的文档块: {len(rows)}")

cnt_kb = defaultdict(int)
cnt_doc = defaultdict(int)
for r in rows:
    cnt_kb[r["kb_id"]] += 1
    cnt_doc[r["doc_id"]] += 1

print("[per-kb] 文档块按 kb_id 分布:")
for k in sorted(cnt_kb):
    print(f"  kb={k}: {cnt_kb[k]}")

print("[per-doc] 文档块按 doc_id 分布:")
for d in sorted(cnt_doc):
    print(f"  doc_id={d}: {cnt_doc[d]}")

# 关键验证：delete_chunks_by_document 的新 filter 是否真能删
print("\n[TEST] 验证删除 filter 可命中（先查后不实删）...")
test_expr = f'kb_id == 6 and doc_id == 22'
hits = client.query(COLL, test_expr, output_fields=["doc_id"], limit=16384)
print(f"  {test_expr} -> {len(hits)} 行 (删除 filter 可命中)")
