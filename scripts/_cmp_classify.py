#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比：新进程(磁盘代码) vs 实时服务(运行进程) 对同一查询的分类域，判断是否加载了旧代码。"""
import sys, os, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="ERROR")
from rag_qa.core.query_classifier import classify_intent

Q = "股票发行注册制改革的主要内容是什么？"
# 1) 磁盘代码（本进程新导入）
disk = classify_intent(Q)
print(f"[磁盘代码] domain={disk.domain} conf={disk.confidence:.4f} hits={disk.keywords_hit}")

# 2) 实时服务
try:
    r = requests.post("http://127.0.0.1:8000/api/v1/chat/",
                      json={"message": Q, "session_id": "cmp", "save_user": False}, timeout=120)
    d = r.json().get("data", r.json())
    print(f"[实时服务] domain={d.get('domain')} conf={d.get('confidence', 0):.4f}")
except Exception as e:
    print(f"[实时服务] 请求失败: {e}")

# 顺带打印磁盘代码里 investment_banking 覆盖规则实际包含的短语
from rag_qa.core.query_classifier import DOMAIN_OVERRIDE_RULES
ib = [ph for ph, dm in DOMAIN_OVERRIDE_RULES if dm == "investment_banking"]
print(f"[磁盘 investment_banking 覆盖短语] {ib}")
print(f"[磁盘是否含 '股票发行注册制' 于短语中] {'股票发行注册制' in sum(ib, ())}")
