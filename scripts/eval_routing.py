#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/eval_routing.py — P3 编排路由准确率评测（门禁：路由准确率 ≥ 90%）。

数据：scripts/route_skill_eval.json（约 218 题，已标 skill 标签）。
方法：对每题跑纯规则路由 `_decide_skill`（离线、确定性、可复现），
输出总准确率 + 各 skill 精确率/召回率 + 混淆矩阵；低于门禁则退出码 1。

用法：
    D:\\FinRag\\.venv\\Scripts\\python.exe scripts/eval_routing.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_qa.orchestrator.route import _decide_skill, SKILLS  # noqa: E402

GATE = 0.90  # 路由准确率门禁

DS_PATH = Path(__file__).resolve().parent / "route_skill_eval.json"


def main() -> int:
    data = json.loads(DS_PATH.read_text(encoding="utf-8"))
    total = len(data)
    correct = 0
    cm: dict[tuple[str, str], int] = defaultdict(int)  # (true, pred)
    misses: list[tuple[str, str, str]] = []

    for it in data:
        q, true = it["question"], it["skill"]
        pred = _decide_skill(q)
        cm[(true, pred)] += 1
        if pred == true:
            correct += 1
        else:
            misses.append((q, true, pred))

    acc = correct / total
    print("=" * 60)
    print(f"路由评测数据集: {total} 题")
    print(f"总准确率: {acc:.2%} (门禁 {GATE:.0%}) | 正确 {correct} / 错误 {total - correct}")
    print("=" * 60)

    # 各 skill 精确率/召回率
    print("\n各 skill 精确率(Precision) / 召回率(Recall):")
    for skill in sorted(SKILLS, key=lambda s: -sum(v for (t, _), v in cm.items() if t == s)):
        tp = cm[(skill, skill)]
        fp = sum(v for (t, p), v in cm.items() if p == skill and t != skill)
        fn = sum(v for (t, _), v in cm.items() if t == skill) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {skill:10s} P={p:.2%}  R={r:.2%}  (支持数={tp + fn})")

    # 混淆矩阵
    print("\n混淆矩阵 (pred \\ true):")
    hdr = "pred\\true |" + "".join(f"{s:>11s}" for s in SKILLS)
    print(hdr)
    for p in SKILLS:
        row = f"{p:>9s}|" + "".join(f"{cm[(t, p)]:>11d}" for t in SKILLS)
        print(row)

    if misses:
        print("\n误路由样本（前 25 条）:")
        for q, t, p in misses[:25]:
            print(f"  [{t}->{p}] {q}")

    ok = acc >= GATE
    print("\n" + ("PASS: 路由准确率达到门禁" if ok else "FAIL: 低于路由准确率门禁"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())