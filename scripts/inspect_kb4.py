import sys
sys.path.insert(0, ".")
from db.milvus import get_milvus_client

client = get_milvus_client()
COLL = "finrag_faq"

# 拉取 kb=4 全部文档块，看 question 前缀与 source_file 分布
rows = client.query(COLL, 'kb_id == 4', output_fields=["doc_id","question","source_file"], limit=16384)
print(f"kb=4 文档块总数: {len(rows)}")

from collections import defaultdict, Counter
by_doc = defaultdict(int)
src = Counter()
prefix = Counter()
for r in rows:
    by_doc[r.get("doc_id")] += 1
    sf = r.get("source_file","")
    src[sf] += 1
    q = r.get("question","")
    if q.startswith("doc_"):
        # 提取 doc_id 段
        try:
            did = q.split("_")[1]
            prefix[f"doc_{did}"] += 1
        except: pass

print("\n按 doc_id 分布:")
for d in sorted(by_doc, key=lambda x:(x is None, x)):
    print(f"  doc_id={d}: {by_doc[d]}")

print("\n按 chunk-id 前缀 (doc_<N>) 分布:")
for p in sorted(prefix):
    print(f"  {p}: {prefix[p]}")

print("\n按 source_file 分布:")
for s,c in src.most_common():
    print(f"  {c:>4}  {s!r}")
