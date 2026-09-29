import sys
sys.path.insert(0, ".")
from collections import defaultdict, Counter
from db.milvus import get_milvus_client

client = get_milvus_client()
COLL = "finrag_faq"
# 取所有 question 以 doc_31_ 开头的块
rows = client.query(COLL, 'question like "doc_31_%"', output_fields=["doc_id","source_file","question"], limit=16384)
print(f"question 以 doc_31_ 开头的块总数: {len(rows)}")
prof = Counter()
for r in rows:
    prof[(r.get("doc_id"), r.get("source_file"))] += 1
print("按 (doc_id, source_file) 分布:")
for k,v in prof.most_common():
    print(f"  doc_id={k[0]!r:>8}  source_file={k[1]!r:>30} -> {v}")
# 是否每个 question 重复几次
qcount = Counter(r["question"] for r in rows)
dups = {q:c for q,c in qcount.items() if c>1}
print(f"\n重复 question 数: {len(dups)}；最大重复次数: {max(qcount.values()) if qcount else 0}")
