# -*- coding: utf-8 -*-
"""P5 集成复验：_build_context 的行情注入是否按预期接线（不加载模型，秒级）。

做法：只把「召回」与「精排」两层打桩，其余全是真实代码路径——
这样能精确验证注入点、拼接顺序、sources 结构与不命中兜底，
而不必支付 BGE-M3 冷加载（约 1 分钟）的代价。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.core.query_classifier import classify_intent  # noqa: E402
from routers import chat as C  # noqa: E402

FAKE_SRC = [{"question": "文档片段", "score": 0.82, "source": "投资者保护.pdf", "source_url": "http://x"}]


def _install_stubs(candidates):
    C._recall_with_strategy = lambda q, d, s: list(candidates)
    C._rerank_context = lambda q, c: ("【问题】文档片段\n【回答】知识库里的制度性内容…", list(FAKE_SRC))


print("=" * 64)
print("A) 行情问句 + 知识库有命中 -> 行情块应拼在 KB 上下文之前")
print("=" * 64)
_install_stubs([{"question": "x", "answer": "y"}])
ctx, srcs = C._build_context("现在的股市行情如何", classify_intent("现在的股市行情如何"))
ok = ctx.startswith("【实时行情数据】") and "【问题】文档片段" in ctx and len(srcs) == 2
ok = ok and srcs[0].get("type") == "realtime_market" and srcs[1]["source"] == "投资者保护.pdf"
# 溯源标注必须跟随真实生效源（不得写死），且带数据时间
label_ok = ("腾讯" in (srcs[0].get("source") or "")) and ("东方财富" not in (srcs[0].get("source") or ""))
stamp_ok = "（20" in (srcs[0].get("question") or "")
ok = ok and label_ok and stamp_ok
print(f"  行情块在最前: {ctx.startswith('【实时行情数据】')}")
print(f"  知识库块保留  : {'【问题】文档片段' in ctx}")
print(f"  sources       : {[s.get('type') or 'kb' for s in srcs]}")
print(f"  溯源标注      : {srcs[0].get('source')} | {srcs[0].get('question')}")
print(f"  标注跟随真实源: {label_ok} | 带数据时间: {stamp_ok}")
print(f"  -> {'✓ 通过' if ok else '✗ 失败'}")

print()
print("=" * 64)
print("B) 行情问句 + 知识库零命中 -> 仍应返回行情上下文（不能答'未找到资料'）")
print("=" * 64)
_install_stubs([])
ctx2, srcs2 = C._build_context("现在的股市行情如何", classify_intent("现在的股市行情如何"))
ok2 = ctx2.startswith("【实时行情数据】") and len(srcs2) == 1 and srcs2[0].get("type") == "realtime_market"
print(f"  context 首行: {ctx2.splitlines()[0] if ctx2 else '(空)'}")
print(f"  sources     : {[s.get('type') or 'kb' for s in srcs2]}")
print(f"  -> {'✓ 通过' if ok2 else '✗ 失败'}")

print()
print("=" * 64)
print("C) 非行情问句 -> 行为必须与升级前完全一致（无行情块）")
print("=" * 64)
_install_stubs([{"question": "x", "answer": "y"}])
ctx3, srcs3 = C._build_context("存款保险最高赔付多少", classify_intent("存款保险最高赔付多少"))
ok3 = "【实时行情数据】" not in ctx3 and len(srcs3) == 1
print(f"  含行情块: {'【实时行情数据】' in ctx3}（应 False）| sources={len(srcs3)}（应 1）")
print(f"  -> {'✓ 通过' if ok3 else '✗ 失败'}")

_install_stubs([])
ctx4, srcs4 = C._build_context("存款保险最高赔付多少", classify_intent("存款保险最高赔付多少"))
ok4 = ctx4 == "" and srcs4 == []
print(f"  非行情+零命中 -> context={ctx4!r} sources={srcs4}（应 空/空）")
print(f"  -> {'✓ 通过' if ok4 else '✗ 失败'}")

print()
print("=" * 64)
print("D) 缓存旁路与 BM25 让路（源码级确认，防止后续被改回去）")
print("=" * 64)
src_text = Path("routers/chat.py").read_text(encoding="utf-8")
checks = [
    ("缓存读旁路", "cached = None if realtime else cache_get(question)"),
    ("BM25 早退让路", "if (not realtime and bm25_best"),
    ("BM25 写缓存旁路", "if not realtime:          # 行情类问句不写缓存"),
    ("老链路写缓存旁路", "if not realtime:              # 行情类问句不写缓存"),
    ("统一入口缓存旁路", 'cacheable = skill_hint in ("rag", "chitchat") and not realtime'),
]
allok = True
for label, needle in checks:
    hit = needle in src_text
    allok &= hit
    print(f"  {'✓' if hit else '✗'} {label}")
print(f"  -> {'✓ 全部就位' if allok else '✗ 有缺失'}")

print()
print("总体:", "✓ 全部通过" if all([ok, ok2, ok3, ok4, allok]) else "✗ 存在失败项")
