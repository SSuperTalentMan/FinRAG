import sys
sys.path.insert(0, ".")
from db.milvus import get_milvus_client

client = get_milvus_client()
COLL = "finrag_faq"

# 精确匹配（非 like），看 doc_31_parent_0_child_0 真实有几行
for q in ['doc_31_parent_0_child_0', 'doc_31_parent_0_child_1', 'doc_31_parent_5_child_0']:
    n = client.query(COLL, f'question == "{q}"', output_fields=["doc_id"], limit=16384)
    print(f'[exact] question=="{q}" -> {len(n)} 行, doc_ids={set(r.get("doc_id") for r in n)}')

# 用 doc_id == 31 精确取全部，统计去重
rows = client.query(COLL, 'doc_id == 31', output_fields=["question"], limit=16384)
print(f'\n[doc_id==31] 总行数={len(rows)}, 唯一 question={len(set(r["question"] for r in rows))}')

# 对照：doc_id == 4 精确
r4 = client.query(COLL, 'doc_id == 4', output_fields=["question"], limit=16384)
print(f'[doc_id==4] 总行数={len(r4)}, 唯一 question={len(set(r["question"] for r in r4))}')
