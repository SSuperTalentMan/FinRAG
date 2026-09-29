#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ingest_blanks.py — 为 FinRag 空白主题批量建库并上传文档（幂等）。

- 按内容主题细分 5 个新知识库；
- 上传 12 个文件中的 10 个（已上传的 2 个自动跳过）；
- 上传时显式传入 domain，使 Milvus chunk 的 category 与知识库 domain 一致，
  保证对话检索（按 intent.domain 过滤）能命中；
- 轮询 /document/processing 直到全部完成，最后校验状态。
"""
import os
import sys
import time

import requests

BASE = "http://localhost:8000/api/v1"
SRC = r"D:\汪欢\Documents\投资补充资料"

# (知识库名, 描述, domain, [相对 SRC 的文件名])
PLAN = [
    ("征信与存款保险", "个人征信查询、信用报告、征信异议处理与存款保险条例", "banking", [
        "banking-存款保险条例.txt",
        "banking-个人信用报告查询渠道及查询流程介绍.txt",
        "banking-个人信用报告介绍.txt",
        "banking-个人征信异议服务流程介绍.txt",
    ]),
    ("会计准则", "企业会计准则、会计准则修订与会计实务问答", "financial_accounting", [
        "财政部会计司有关负责人就修订印发答记者问.txt",
        "企业会计准则.txt",
    ]),
    ("数字人民币", "数字人民币（e-CNY）机制、管理体系与应用场景", "fintech", [
        "数字人民币.txt",
    ]),
    ("金融科技", "金融科技发展规划与金融数字化转型", "fintech", [
        "金融科技分展规划.txt",
    ]),
    ("投资者保护（防非反诈）", "防范非法证券、异常波动风险提示与上市公司公告价值判断", "stock_market", [
        "谨防异常波动股票的投资风险.txt",
        "投资者如何判断上市公司公告中的价值和风险.md",
    ]),
]

# 已上传、需跳过的文件（来自既有库：banking 库 id=1、投资教育库 id=4）
ALREADY_UPLOADED = {
    "banking-常见问题解答.txt",          # 已在「银行业务」库 (kb_id=1)
    "投资者教育文章.pdf",                # 已在「投资教育」库 (kb_id=4)
}


def get_kb_map():
    r = requests.get(f"{BASE}/knowledge_base/list", timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    return {kb["name"]: kb["id"] for kb in data}


def create_kb(name, desc, domain):
    r = requests.post(f"{BASE}/knowledge_base/", json={
        "name": name, "description": desc, "domain": domain,
    }, timeout=10)
    if r.status_code == 200 and r.json().get("success"):
        return r.json()["data"]["id"]
    raise RuntimeError(f"创建知识库失败 {name}: {r.status_code} {r.text[:200]}")


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


def main():
    kb_map = get_kb_map()
    print(f"[init] 当前已有知识库: {kb_map}")

    # 1) 幂等创建知识库（按名称复用）
    plan_ids = {}
    for name, desc, domain, _files in PLAN:
        if name in kb_map:
            kb_id = kb_map[name]
            print(f"[kb] 复用已有知识库: {name} (id={kb_id})")
        else:
            kb_id = create_kb(name, desc, domain)
            print(f"[kb] 新建知识库: {name} (id={kb_id}, domain={domain})")
        plan_ids[name] = (kb_id, domain)

    # 2) 上传文件
    skipped = []
    triggered = []  # (name, doc_id, fname)
    for name, _desc, domain, files in PLAN:
        kb_id, _ = plan_ids[name]
        for fname in files:
            if fname in ALREADY_UPLOADED:
                print(f"[skip] 已上传，跳过: {fname}")
                skipped.append(fname)
                continue
            fp = os.path.join(SRC, fname)
            if not os.path.exists(fp):
                print(f"[warn] 文件不存在，跳过: {fp}")
                continue
            try:
                code, body = upload(kb_id, fp, domain)
            except Exception as e:
                print(f"[err] 上传异常 {fname}: {e}")
                continue
            if code == 409:
                print(f"[skip] 同名已存在(409)，跳过: {fname}")
                skipped.append(fname)
                continue
            if code != 200:
                print(f"[err] 上传失败 {fname}: {code} {body[:200]}")
                continue
            doc_id = __import__("json").loads(body)["data"]["doc_id"]
            print(f"[up] 已触发上传: {fname} -> kb_id={kb_id} doc_id={doc_id} domain={domain}")
            triggered.append((name, doc_id, fname))

    print(f"\n[sum] 触发上传 {len(triggered)} 个，跳过 {len(skipped)} 个: {skipped}")

    # 3) 轮询直到处理队列清空
    print("\n[wait] 等待后台解析/向量化完成（串行队列，请耐心等待）...")
    start = time.time()
    last_phase = {}
    while time.time() - start < 1800:
        items = processing_items()
        if not items:
            time.sleep(3)
            if not processing_items():
                print("[wait] 处理队列已清空，所有任务结束。")
                break
            continue
        for it in items:
            key = it["doc_id"]
            last_phase[key] = (it["filename"], it["phase"], it["done"], it["total"])
        line = "; ".join(
            f"{v[0]} {v[1]} {v[2]}/{v[3]}" for v in last_phase.values()
        )
        print(f"  ... {line}", flush=True)
        time.sleep(8)
    else:
        print("[warn] 等待超时（30min），请稍后手动检查。")

    # 4) 校验各库文档状态
    print("\n[verify] 校验各知识库文档状态：")
    for name, _desc, domain, _files in PLAN:
        kb_id, _ = plan_ids[name]
        r = requests.get(f"{BASE}/document/list", params={"kb_id": kb_id}, timeout=10)
        docs = r.json()["data"]["items"]
        statuses = {}
        for d in docs:
            statuses[d["status"]] = statuses.get(d["status"], 0) + 1
        print(f"  - {name} (kb_id={kb_id}): 共 {len(docs)} 个文档, 状态={statuses}")
        for d in docs:
            print(f"      [{d['status']}] {d['filename']} (chunks={d.get('chunk_count')})")

    # 5) 全局知识库列表
    print("\n[final] 知识库列表：")
    kb_map2 = get_kb_map()
    r = requests.get(f"{BASE}/knowledge_base/list", timeout=10).json()["data"]
    for kb in r:
        print(f"  id={kb['id']} {kb['name']} domain={kb['domain']} doc_count={kb['doc_count']}")


if __name__ == "__main__":
    main()
