import sys, pymysql
sys.path.insert(0, ".")
from config import get_config

cfg = get_config()
m = cfg.mysql
conn = pymysql.connect(host=m.host, port=m.port, user=m.user,
                       password=m.password, database=m.database)
cur = conn.cursor(pymysql.cursors.DictCursor)
cur.execute("SHOW COLUMNS FROM documents")
cols = [r["Field"] for r in cur.fetchall()]
print("documents 列:", cols)
cur.execute(f"SELECT {', '.join(c for c in cols if c in ('id','kb_id','status','chunk_count','filename','file_name','name','title','source_file','deleted'))} FROM documents ORDER BY id")
rows = cur.fetchall()
print(f"documents 总数: {len(rows)}")
for r in rows:
    print("  ", r)
conn.close()
