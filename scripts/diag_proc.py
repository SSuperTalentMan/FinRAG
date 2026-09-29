import sys, os, tempfile, shutil
sys.path.insert(0, ".")
from config import get_config
from rag_qa.core.document_processor import process_documents

SRC = r"D:/汪欢/Documents/投资补充资料/投资者教育文章.pdf"
cfg = get_config()
tmp = tempfile.mkdtemp(prefix="diag_")
shutil.copy2(SRC, os.path.join(tmp, "投资者教育文章.pdf"))
chunks = process_documents(tmp,
    parent_chunk_size=cfg.retrieval.parent_chunk_size,
    child_chunk_size=cfg.retrieval.child_chunk_size,
    chunk_overlap=cfg.retrieval.chunk_overlap,
    doc_id=31)
ids = [c.metadata.get("id","") for c in chunks]
from collections import Counter
print(f"process_documents 返回子块数: {len(chunks)}")
print(f"唯一 id 数: {len(set(ids))}")
print("id 频次 Top10:")
for i,c in Counter(ids).most_common(10):
    print(f"  {c:>4}  {i!r}")
shutil.rmtree(tmp, ignore_errors=True)
