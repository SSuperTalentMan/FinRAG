# -*- coding: utf-8 -*-
"""临时运维脚本：端到端验证新建知识库内容是否可被检索命中（只读）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config  # noqa: E402
from db.milvus import ensure_collection, get_milvus_client, search_milvus  # noqa: E402
from rag_qa.core.query_classifier import classify_intent  # noqa: E402
from services.embedding import encode_query_dense_sparse  # noqa: E402

QUESTIONS = [
    "个人信用报告可以在哪里查询？查询流程是什么？",
    "存款保险的偿付限额是多少？",
    "存款保险条例规定最高偿付限额是多少？",
    "什么是存款保险？哪些存款受保障？",
    "企业会计准则中收入确认的原则是什么？",
    "数字人民币是什么？和微信支付有什么区别？",
    "金融科技发展的主要目标是什么？",
    "如何防范异常波动股票的投资风险？",
    "投资者如何判断上市公司公告中的投资价值？",
    "个人征信有异议时怎么申请处理？",
]


def main():
    cfg = get_config()
    client = get_milvus_client()
    ensure_collection(client)

    for q in QUESTIONS:
        intent = classify_intent(q)
        qd, qs = encode_query_dense_sparse(q)
        dom = intent.domain if intent.domain != "general" else None
        hits = search_milvus(client, qd, qs, k=cfg.retrieval.retrieval_k, domain=dom)
        print("=" * 96)
        print(f"Q: {q}")
        print(f"   意图域={intent.domain} (conf={intent.confidence:.3f}) 过滤={dom} 命中={len(hits)}")
        for h in hits[:3]:
            ans = h["answer"].replace("\n", " ")[:90]
            print(f"   [{h['category']}|{h['score']:.4f}] {ans}")
    print("=" * 96)


if __name__ == "__main__":
    main()
