import sys, re
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

# 取全部 doc_ 前缀块（按 question 前缀，不依赖 doc_id 字段）
rows = client.query("finrag_faq", 'question like "doc_%"', output_fields=["question","doc_id"], limit=16384)
print(f"question 以 doc_ 开头的块: {len(rows)}")
pat = re.compile(r"^doc_(\d+)_")
bad = Counter()
total_bad = 0
examples = []
for r in rows:
    q = r.get("question","")
    mm = pat.match(q)
    if not mm:
        continue
    did = int(mm.group(1))
    if did not in valid:
        bad[did] += 1
        total_bad += 1
        if len(examples) < 5:
            examples.append((did, q, r.get("doc_id")))
print(f"属于 MySQL 不存在文档的块（按 question 前缀）: {total_bad}")
print("按 doc_id(前缀) 分布:", dict(bad))
print("样例:", examples)
