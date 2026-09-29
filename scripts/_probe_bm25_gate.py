# -*- coding: utf-8 -*-
"""校准 BM25 早退门控阈值。

背景
----
`services/bm25.py` 的 `softmax_score` 是在**全部 FAQ 候选**上做 softmax，衡量的是
"这条比其他 FAQ 领先多少"（相对分数），而非"它和问题是否真的匹配"（绝对相关性）。
只要某条 FAQ 的 BM25 原始分明显领先，softmax 就会逼近 1.0。

实测事故：
    「股票发行注册制改革的主要内容是什么？」
      → BM25 命中 FAQ「债券注册制改革全面落地」softmax=0.985 ≥ 0.95 → 早退
      → 返回债券注册制答案，把 doc_4（股票发行注册制投教问答）完全挡在检索之外。

因此单纯调高 softmax 阈值治不了本（0.985 已接近 1.0）。正确做法是在早退前加一道
**绝对相关性校验**（BGE-Reranker cross-encoder 打分）。本脚本用于校准该门控阈值：
同时打印正样本（FAQ 原问题/改写，本应早退）与负样本（历史误命中，本不该早退）的
softmax 与 reranker 分数，找出可安全区分两者的阈值。

用法：
    .venv/Scripts/python.exe scripts/_probe_bm25_gate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.mysql import get_all_qa_for_bm25  # noqa: E402
from services.bm25 import get_bm25_retriever  # noqa: E402
from services.reranker import get_reranker, rerank_text  # noqa: E402

# 负样本：本不该早退（BM25 语义沾边但答非所问，真答案在文档块里）
NEG = [
    "股票发行注册制改革的主要内容是什么？",   # 事故案例：误命中"债券注册制改革"
    "个人信用报告可以在哪里查询？",           # 历史案例：误命中"减免征信报告查询费用"
    "存款保险的偿付限额是多少？",             # 历史案例：误命中"引入存款保险的建议理由"
    "投资者教育包括哪些内容？",
    "注册制下发行人信息披露有哪些要求？",
]

# 正样本：本应早退（直接取 FAQ 原问题，以及口语化改写）
qa = get_all_qa_for_bm25() or []
POS_EXACT = [d["question"] for d in qa[:6]]
POS_PARAPHRASE = [(d["question"].rstrip("？?") + "呢？") for d in qa[:3]]

bm25 = get_bm25_retriever()
model = get_reranker()


def probe(tag: str, queries: list[str]) -> list[tuple[float, float]]:
    """打印每条 query 的 BM25 命中及 reranker 绝对相关分，返回 (softmax, rr) 列表。"""
    out = []
    print("=" * 116)
    print(f"### {tag}")
    for q in queries:
        # 与 routers/chat.py Step 2 完全一致：domain=None、threshold=0.75
        best = bm25.get_best_match(q, threshold=0.75)
        if not best:
            print(f"  [无BM25命中] {q[:44]}")
            continue
        rr = float(model.predict([(q, rerank_text(best))])[0])
        sm = best["softmax_score"]
        flag = "早退" if sm >= 0.95 else "走管线"
        print(f"  softmax={sm:.4f} 原始={best['score']:>6.2f} rerank={rr:>7.4f} [{flag}] "
              f"Q={q[:34]} -> FAQ={best['question'][:34]}")
        out.append((sm, rr))
    return out


neg = probe("负样本（本不该早退）", NEG)
pos1 = probe("正样本 A：FAQ 原问题（应早退）", POS_EXACT)
pos2 = probe("正样本 B：FAQ 改写（应早退）", POS_PARAPHRASE)

pos = pos1 + pos2
print("=" * 116)
print("### 阈值可分性分析（仅统计 softmax≥0.95 即会触发早退的样本）")
neg_gate = [rr for sm, rr in neg if sm >= 0.95]
pos_gate = [rr for sm, rr in pos if sm >= 0.95]
if neg_gate:
    print(f"  会误早退的负样本 rerank 分数: {[round(x, 4) for x in neg_gate]}  max={max(neg_gate):.4f}")
else:
    print("  无负样本会触发早退")
if pos_gate:
    print(f"  应早退的正样本 rerank 分数: {[round(x, 4) for x in pos_gate]}  min={min(pos_gate):.4f}")
if neg_gate and pos_gate:
    print(f"  => 建议门控阈值取区间 ({max(neg_gate):.4f}, {min(pos_gate):.4f}) 内的值")
    if max(neg_gate) >= min(pos_gate):
        print("  !! 两者重叠，reranker 单独不足以区分，需结合其他信号")
