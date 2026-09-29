import sys
sys.path.insert(0, ".")
from config import get_config
import pymysql
cfg = get_config(); m = cfg.mysql
conn = pymysql.connect(host=m.host, port=m.port, user=m.user, password=m.password, database=m.database)
cur = conn.cursor()
cur.execute("UPDATE documents SET chunk_count=490 WHERE id=4")
conn.commit()
cur.execute("SELECT id, filename, chunk_count, status FROM documents WHERE id=4")
print("updated:", cur.fetchone())
conn.close()
