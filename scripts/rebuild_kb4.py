#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_kb4.py — 仅重建 kb=4（投资教育）中能找到源文件的文档，打破 chunk id 跨文件重复。

背景：
  kb=4 有 2 个 PDF，早年单文件上传时文件序号都=0，导致两文件的 chunk id
  （doc_0_parent_j_child_k）完全互撞，检索按 question 去重会丢块/串内容。
  修复后的 process_documents 用全局唯一 doc_id 构造 chunk id，故重建即可消除重复。
  其中「全面实行股票发行注册制改革投教问答.pdf」源文件已丢失（无法先删后传），
  仅重建「投资者教育文章.pdf」（源在投资补充资料），重建后其获得唯一 doc_id，
  与另一文件不再同名碰撞，跨文件重复问题消除；另一文件保留不动。

用法：服务已重启并加载新代码后执行：
    .venv\Scripts\python.exe scripts\rebuild_kb4.py
"""
import json
import os
import sys
import time

import requests

BASE = "http://localhost:8000/api/v1"
SRC = r"D:\汪欢\Documents\投资补充资料"
KB_ID = 4
DOMAIN = "investment_banking"
FNAME = "投资者教育文章.pdf"


def wait_empty(timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/document/processing", timeout=10)
        r.raise_for_status()
        items = r.json()["data"]["items"]
        if not items:
            time.sleep(2)
            if not requests.get(f"{BASE}/document/processing", timeout=10).json()["data"]["items"]:
                return True
            continue
        line = "; ".join(f"{it['filename']} {it['phase']} {it['done']}/{it['total']}" for it in items)
        print(f"  ... {line}", flush=True)
        time.sleep(6)
    return False


def main():
    fp = os.path.join(SRC, FNAME)
    if not os.path.exists(fp):
        print(f"[abort] 源文件不存在: {fp}")
        return
    # 1) 删除旧文档（精确按 doc_id）
    r = requests.get(f"{BASE}/document/list", params={"kb_id": KB_ID}, timeout=10)
    r.raise_for_status()
    for d in r.json()["data"]["items"]:
        if d["filename"] == FNAME and d["status"] != "deleted":
            code = requests.delete(f"{BASE}/document/{d['id']}", timeout=10).status_code
            print(f"[del] {FNAME} doc_id={d['id']} -> {code}")
    # 2) 重新上传（新代码生成唯一 chunk id）
    with open(fp, "rb") as f:
        r = requests.post(
            f"{BASE}/document/upload",
            data={"kb_id": str(KB_ID), "domain": DOMAIN},
            files={"file": (FNAME, f, "application/octet-stream")},
            timeout=30,
        )
    if r.status_code != 200:
        print(f"[err] 上传失败: {r.status_code} {r.text[:200]}")
        return
    doc_id = json.loads(r.text)["data"]["doc_id"]
    print(f"[up] 已触发重传: {FNAME} -> doc_id={doc_id}")
    if not wait_empty():
        print("[warn] 等待超时")
    # 3) 验证
    r = requests.get(f"{BASE}/document/list", params={"kb_id": KB_ID}, timeout=10)
    for d in r.json()["data"]["items"]:
        print(f"  [{d['status']}] {d['filename']} chunks={d.get('chunk_count')}")
    print("\n[done] 重建完成。建议跑 scripts/verify_kb_retrieval.py 或 check_chunk_ids.py 复验。")


if __name__ == "__main__":
    main()
