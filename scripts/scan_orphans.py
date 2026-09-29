import sys
sys.path.insert(0, ".")
from collections import Counter
from db.milvus import get_milvus_client
from config import get_config
import pymysql

client = get_milvus_client()
cfg = get_config(); m = cfg.mysql
conn = pymysql.connect(host=m.host, port=m.port, user=m.user, password=m.password, database=m.database)
cur = conn.cursor()
cur.execute("SELECT id FROM documents")
valid = set(r[0] for r in cur.fetchall())
conn.close()

# 取全部文档块（含 FAQ 之外的），统计 doc_id 分布
rows = client.query(client, 'doc_id > 0', output_fields=["doc_id"], limit=16384) if False else \
       client.query("finrag_faq", 'doc_id > 0', output_fields=["doc_id"], limit=16384)
cnt = Counter(r.get("doc_id") for r in rows)
print(f"Milvus 含 doc_id 的块总数: {len(rows)}")
print("doc_id 分布（含 MySQL 不存在的）:")
orphans = {}
for d in sorted(cnt):
    tag = "" if d in valid else "  <== ORPHAN(MySQL无此文档)"
    if d not in valid:
        orphans[d] = cnt[d]
    print(f"  doc_id={d}: {cnt[d]}{tag}")
print(f"\n孤儿 doc_id 集合: {orphans}")
print(f"孤儿块总数: {sum(orphans.values())}")
