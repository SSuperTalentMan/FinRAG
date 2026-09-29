#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reindex_kb_docs.py — 重建「空白主题」5 个知识库的文档向量（修复 chunk id 跨文件重复）。

背景：
  早期上传走 process_documents 时，单文件上传的文件序号 i 恒为 0，
  导致同知识库内多个文件生成的 chunk id（doc_0_parent_j_child_k）完全相同，
  检索去重按 question 会丢块/串内容。修复后的 process_documents 用全局唯一的
  doc_id 构造 chunk id（doc_{doc_id}_parent_j_child_k），保证唯一。

用法（重要）：
  本脚本依赖修复后的代码生效，请先在运行 main.py 的终端 Ctrl+C 重启服务，再执行：
      .venv\Scripts\python.exe scripts\reindex_kb_docs.py
  脚本会：对每个文件先 DELETE 旧文档（清 MySQL 记录 + Milvus 向量），再重新
  POST /document/upload，最后轮询直到全部 indexed，并打印各库状态。

  仅重建这 10 个文档；banking-常见问题解答.txt / 投资者教育文章.pdf 不在范围内
  （它们在其他库，且 id 唯一性问题不影响单文件库）。
"""
import json
import os
import sys
import time

import requests

BASE = "http://localhost:8000/api/v1"
SRC = r"D:\汪欢\Documents\投资补充资料"

# (知识库名, domain, [相对 SRC 的文件名]) —— 与 ingest_blanks.py 的 PLAN 对应
PLAN = [
    ("征信与存款保险", "banking", [
        "banking-存款保险条例.txt",
        "banking-个人信用报告查询渠道及查询流程介绍.txt",
        "banking-个人信用报告介绍.txt",
        "banking-个人征信异议服务流程介绍.txt",
    ]),
    ("会计准则", "financial_accounting", [
        "财政部会计司有关负责人就修订印发答记者问.txt",
        "企业会计准则.txt",
    ]),
    ("数字人民币", "fintech", [
        "数字人民币.txt",
    ]),
    ("金融科技", "fintech", [
        "金融科技分展规划.txt",
    ]),
    ("投资者保护（防非反诈）", "stock_market", [
        "谨防异常波动股票的投资风险.txt",
        "投资者如何判断上市公司公告中的价值和风险.md",
    ]),
]


def get_kb_map():
    r = requests.get(f"{BASE}/knowledge_base/list", timeout=10)
    r.raise_for_status()
    return {kb["name"]: kb["id"] for kb in r.json()["data"]}


def find_doc_id(kb_id, fname):
    r = requests.get(f"{BASE}/document/list", params={"kb_id": kb_id}, timeout=10)
    r.raise_for_status()
    for d in r.json()["data"]["items"]:
        if d["filename"] == fname and d["status"] != "deleted":
            return d["id"]
    return None


def delete_doc(doc_id):
    r = requests.delete(f"{BASE}/document/{doc_id}", timeout=10)
    return r.status_code, (r.json().get("message") if r.content else "")


def upload(kb_id, filepath, domain):
    fname = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{BASE}/document/upload",
            data={"kb_id": str(kb_id), "domain": domain},
            files={"file": (fname, f, "application/octet-stream")},
            timeout=30,
        )
    return r.status_code, r.text


def processing_items():
    r = requests.get(f"{BASE}/document/processing", timeout=10)
    r.raise_for_status()
    return r.json()["data"]["items"]


def wait_empty(timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        items = processing_items()
        if not items:
            time.sleep(2)
            if not processing_items():
                return True
            continue
        line = "; ".join(f"{it['filename']} {it['phase']} {it['done']}/{it['total']}" for it in items)
        print(f"  ... {line}", flush=True)
        time.sleep(6)
    return False


def main():
    kb_map = get_kb_map()
    for name, domain, files in PLAN:
        kb_id = kb_map.get(name)
        if kb_id is None:
            print(f"[skip] 知识库不存在: {name}")
            continue
        print(f"\n=== 重建知识库: {name} (kb_id={kb_id}, domain={domain}) ===")
        for fname in files:
            fp = os.path.join(SRC, fname)
            if not os.path.exists(fp):
                print(f"[warn] 文件不存在，跳过: {fp}")
                continue
            # 1) 删除旧文档（清向量 + 记录）
            old_id = find_doc_id(kb_id, fname)
            if old_id is not None:
                code, msg = delete_doc(old_id)
                print(f"[del] {fname} doc_id={old_id} -> {code} {msg}")
            else:
                print(f"[del] {fname} 未找到旧记录，直接上传")
            # 2) 重新上传（修复后代码生成唯一 chunk id）
            try:
                code, body = upload(kb_id, fp, domain)
            except Exception as e:
                print(f"[err] 上传异常 {fname}: {e}")
                continue
            if code != 200:
                print(f"[err] 上传失败 {fname}: {code} {body[:200]}")
                continue
            doc_id = json.loads(body)["data"]["doc_id"]
            print(f"[up] 已触发重传: {fname} -> doc_id={doc_id}")
            # 3) 等待该文件处理完（串行队列，逐个来）
            if not wait_empty():
                print(f"[warn] 等待超时: {fname}")

    print("\n[verify] 重建后各库文档状态：")
    for name, domain, files in PLAN:
        kb_id = kb_map.get(name)
        if kb_id is None:
            continue
        r = requests.get(f"{BASE}/document/list", params={"kb_id": kb_id}, timeout=10)
        docs = r.json()["data"]["items"]
        print(f"  - {name} (kb_id={kb_id}): {len(docs)} 个文档")
        for d in docs:
            print(f"      [{d['status']}] {d['filename']} chunks={d.get('chunk_count')}")
    print("\n[done] 重建完成。建议跑 scripts/verify_chat_api.py 复验检索。")


if __name__ == "__main__":
    main()
