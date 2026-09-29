# -*- coding: utf-8 -*-
"""回归验证：BM25 早退绝对相关性门控（routers/chat.py::_bm25_early_gate_passed）。

直接调用生产函数 `routers.chat.ask_question`（与 HTTP 接口同一份代码路径），
因此无需重启服务即可验证磁盘代码的正确性。

验证点：
  1. 「股票发行注册制改革的主要内容是什么？」不再被 FAQ「债券注册制改革全面落地」
     早退截获，而应走完整管线并召回 doc_4（股票发行注册制投教问答）。
  2. 原本正确的用例（FAQ 精确命中早退、其他领域文档召回）不被门控破坏。

用法：
    .venv/Scripts/python.exe scripts/_diag_bm25_gate_e2e.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.mysql import get_all_qa_for_bm25  # noqa: E402
from db.redis import clear_qa_cache  # noqa: E402
from models import ChatRequest  # noqa: E402
from routers.chat import ask_question  # noqa: E402

# 清除问答缓存：早退路径会 cache_set，旧的错误答案会掩盖修复效果
try:
    n = clear_qa_cache()
    print(f"[准备] 已清除问答缓存 {n} 条\n")
except Exception as e:
    print(f"[准备] 清缓存失败（继续）: {e}\n")

# (问题, 期望命中的内容标识, 说明)
CASES = [
    ("股票发行注册制改革的主要内容是什么？", "doc_4",
     "事故用例：此前被『债券注册制改革』FAQ 早退截获"),
    ("注册制下发行人信息披露有哪些要求？", "doc_4", "对照：本来就正确"),
    ("投资者教育包括哪些内容？", "doc_31", "对照：本来就正确"),
    ("存款保险的偿付限额是多少？", "doc_21", "对照：softmax 0.912 未达早退线"),
    ("如何防范异常波动股票的投资风险？", "doc_29", "对照：其他领域不受影响"),
]

# 正样本：FAQ 原问题应仍然早退（确认门控没破坏精确命中的快速通道）
qa = get_all_qa_for_bm25() or []
FAQ_EXACT = [d["question"] for d in qa[:2]]

passed = 0
print("=" * 108)
print("### 主用例：门控是否让文档块正常召回")
for q, want, note in CASES:
    resp = ask_question(ChatRequest(message=q))
    d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    srcs = [s.get("question", "") for s in (d.get("sources") or [])]
    hit = [s for s in srcs if s.startswith(want + "_")]
    ok = bool(hit)
    passed += ok
    print("-" * 108)
    print(f"Q: {q}")
    print(f"   {note}")
    print(f"   domain={d.get('domain')} conf={d.get('confidence'):.3f} 召回={len(srcs)} "
          f"期望{want}命中={'是' if ok else '否'}({len(hit)}条)")
    for s in srcs[:3]:
        print(f"     - {s[:70]}")
    print(f"   答案: {str(d.get('answer'))[:150]}")

print("=" * 108)
print("### 反向用例：FAQ 精确命中应仍走早退快速通道（sources 恰为 1 条 FAQ 原问题）")
gate_ok = 0
for q in FAQ_EXACT:
    resp = ask_question(ChatRequest(message=q))
    d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    srcs = [s.get("question", "") for s in (d.get("sources") or [])]
    early = len(srcs) == 1 and srcs[0] == q
    gate_ok += early
    print(f"  [{'早退OK' if early else '未早退'}] conf={d.get('confidence'):.4f} 召回={len(srcs)} Q={q[:46]}")

print("=" * 108)
print(f"结论：主用例 {passed}/{len(CASES)} 召回期望文档；FAQ 精确命中早退 {gate_ok}/{len(FAQ_EXACT)} 保持正常")
