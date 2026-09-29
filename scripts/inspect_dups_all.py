import sys
sys.path.insert(0, ".")
from collections import defaultdict, Counter
from db.milvus import get_milvus_client

client = get_milvus_client()
COLL = "finrag_faq"
rows = client.query(COLL, 'doc_id > 0', output_fields=["doc_id","question"], limit=16384)
print(f"doc_id>0 块总数: {len(rows)}")
qmap = defaultdict(list)
for r in rows:
    qmap[r["question"]].append(r.get("doc_id"))
dups = {q:d for q,d in qmap.items() if len(d)>1}
print(f"重复 question 数: {len(dups)}")
# 重复块按 doc_id 配对分布
pair = Counter()
for q,d in dups.items():
    key = tuple(sorted(set(d)))
    pair[key] += 1
print("重复块 (doc_id 集合) 分布:")
for k,v in pair.most_common(10):
    print(f"  doc_ids={k} -> {v} 个重复 question")
# 抽取一个重复样例
sample = next(iter(dups))
print(f"\n样例重复 question={sample!r} 出现在 doc_id: {dups[sample]}")
