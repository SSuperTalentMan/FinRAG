#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/verify_gov_ingest.py — 验证 gov.cn 政策解读数据是否可被检索命中

对若干典型金融政策问题，走真实检索链路：
  意图分类 → BGE-M3 编码 → Milvus 领域过滤检索，
确认中国政府网来源的数据出现在 Top-K，并打出命中分数。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
_cfg = get_config()

from rag_qa.core.query_classifier import get_classifier
from db.milvus import get_milvus_client, search_milvus, ensure_collection
from services.embedding import encode_query_dense_sparse

# 选取覆盖多领域的政策类问题（应命中 gov.cn 来源）
TEST_QUERIES = [
    "数字人民币试点进展如何",
    "个人养老金怎么参加",
    "防范化解地方债务风险有哪些举措",
    "货币政策如何支持实体经济",
    "上市公司分红政策有什么新规",
    "商业银行资本充足率监管要求",
    "科创板注册制改革方向",
    "普惠金融对小微企业的支持政策",
]


def main():
    import json as _json
    from pathlib import Path as _P
    # 预载 gov 来源问题集合，用于判断命中（Milvus 实体未存 source_name，
    # 故以问题文本匹配 gov_qa.jsonl 中的记录）
    gov_questions = set()
    for line in open(_P(PROJECT_ROOT) / "data" / "crawled" / "gov_qa.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        gov_questions.add(o.get("question", ""))
        gov_questions.add(o.get("source_name", ""))

    def is_gov(hit_q: str) -> bool:
        if not hit_q:
            return False
        for g in gov_questions:
            if g and (g in hit_q or hit_q in g):
                return True
        return False

    clf = get_classifier()
    client = get_milvus_client()
    ensure_collection(client)

    for q in TEST_QUERIES:
        domain, conf, _ = clf.predict(q)
        print(f"\n问题: {q}")
        print(f"  意图分类 -> domain={domain} conf={conf:.2f}")
        dense, sparse = encode_query_dense_sparse(q)
        hits = search_milvus(client, dense, sparse, k=5,
                              domain=domain if domain != "general" else None)
        gov_hit = False
        for h in hits[:5]:
            hit_q = h.get("question", "")
            tag = "★GOV" if is_gov(hit_q) else "    "
            if tag.startswith("★"):
                gov_hit = True
            print(f"    {tag} score={h['score']:.3f} | {hit_q[:46]}")
        print(f"  -> {'命中 gov.cn ✅' if gov_hit else '未见 gov.cn（意图被分到其他域，未检索该域）'}")


if __name__ == "__main__":
    main()
