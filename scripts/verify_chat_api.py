# -*- coding: utf-8 -*-
"""
重启后自检脚本：通过真实 HTTP /chat 接口验证问答链路。

与 verify_kb_retrieval.py 的区别：
- verify_kb_retrieval.py 在本进程内直接调函数，测的是「磁盘上的代码」；
- 本脚本走 HTTP，测的是「当前运行中的服务」，用来确认重启后新代码是否真的生效。

用法：
    .venv/Scripts/python.exe scripts/verify_chat_api.py

说明：每条用例提问前会清掉该问题的 Redis 缓存，避免命中旧答案导致误判。
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.redis import cache_delete  # noqa: E402

BASE = "http://127.0.0.1:8000/api/v1"

# (问题, 期望在召回来源中出现的关键字)
CASES = [
    ("存款保险的偿付限额是多少？", ["存款保险"]),
    ("个人信用报告可以在哪里查询？", ["信用报告"]),
    ("个人征信有异议时怎么申请处理？", ["征信"]),
    ("企业会计准则中收入确认的原则是什么？", ["准则"]),
    ("数字人民币迎来了什么重大调整？", ["数字人民币"]),
    ("如何防范异常波动股票的投资风险？", ["异常波动"]),
    # kb=4 投资教育（doc_4 注册制投教问答 / doc_31 投资者教育文章）
    # —— 验证意图分类覆盖规则把「注册制/投资者教育」正确路由到 investment_banking 域
    ("注册制下发行人信息披露有哪些要求？", ["doc_4_"]),
    ("股票发行注册制改革的主要内容是什么？", ["doc_4_"]),
    ("投资者教育包括哪些内容？", ["doc_31_"]),
]


def main():
    ok_count = 0
    for q, expect_kws in CASES:
        cache_delete(q)  # 清缓存，确保走完整检索链路
        try:
            r = requests.post(
                f"{BASE}/chat/",
                json={"message": q, "session_id": "verify-script", "save_user": False},
                timeout=180,
            )
            r.raise_for_status()
            payload = r.json()
            # /chat/ 非流式接口 response_model=ChatResponse，响应体无 data 包裹层；
            # 这里兼容 {success, data} 与直接返回两种结构。
            data = payload.get("data", payload) if isinstance(payload, dict) else {}
            if "answer" not in data:
                raise ValueError(f"响应结构异常: {str(payload)[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"Q: {q}\n   [请求失败] {e}\n")
            continue

        sources = data.get("sources", [])
        src_text = " || ".join(s.get("question", "") for s in sources)
        hit = any(kw in src_text for kw in expect_kws)
        ok_count += 1 if hit else 0

        print("=" * 92)
        print(f"Q: {q}")
        print(f"   domain={data.get('domain')} conf={data.get('confidence', 0):.3f} "
              f"召回={len(sources)} 命中期望内容={'是' if hit else '否'}")
        for s in sources[:3]:
            print(f"   - [{s.get('source', '')}|{s.get('score', 0):.4f}] {s.get('question', '')[:70]}")
        ans = data.get("answer", "").replace("\n", " ")
        print(f"   答案: {ans[:160]}")

    print("=" * 92)
    print(f"自检完成：{ok_count}/{len(CASES)} 条用例召回了期望的新库内容")


if __name__ == "__main__":
    main()
