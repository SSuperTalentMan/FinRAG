#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/crawl_gov_targeted.py — gov.cn 定向补抓（弥补全量爬取的 max 上限缺口）

针对端到端验证暴露的两个弱查询——「个人养老金怎么参加」「上市公司分红有哪些规定」——
原全量爬取按时间倒序取前 1300 页，把较早的 个人养老金 / 分红 政策解读挤到了上限之外。
本脚本按关键词（个人养老金 / 养老金 / 分红 / 现金分红）从同一官方列表 JSON 中定向补抓，
复用 crawl_gov.py 的正文抽取与问答解析逻辑，增量追加到 data/crawled/gov_qa.jsonl
（按 source_url 断点续爬 + 去重），随后用 ingest_crawled.py --only gov 重新入库即可。

用法：
    python scripts/crawl_gov_targeted.py --max 80 --delay 1.0
"""
import json
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from loguru import logger

# 复用既有爬虫的抽取与落盘逻辑，避免重复实现
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crawl_gov import (  # noqa: E402
    fetch, robots_ok, strip_html, extract_main_html, extract_qa_pairs,
    extract_qa_paragraphs, clean_body, load_done, LIST_URL, OUT_DIR,
)

CST = timezone(timedelta(hours=8))

# 定向关键词：直击两个弱查询的主题
TARGET_KEYWORDS = ("个人养老金", "养老金", "分红", "现金分红", "股息")


def main():
    ap = argparse.ArgumentParser(description="gov.cn 定向补抓（个人养老金/分红等）")
    ap.add_argument("--max", type=int, default=80, help="本次最多抓取的详情页数")
    ap.add_argument("--delay", type=float, default=1.0, help="请求间隔（秒）")
    args = ap.parse_args()

    if not robots_ok():
        logger.error("robots.txt 禁止，终止")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "gov_qa.jsonl"
    done_urls, base_n = load_done(out)
    if done_urls:
        logger.info(f"[GOV-T] 断点续爬：已有 {base_n} 条 / {len(done_urls)} 页面，跳过")

    raw = fetch(LIST_URL, 0)
    if not raw:
        logger.error("无法获取列表 JSON")
        return
    items = json.loads(raw)
    logger.info(f"[GOV-T] 列表共 {len(items)} 条")

    targets = []
    for it in items:
        title = (it.get("TITLE") or "").strip()
        if not title:
            continue
        if not any(k in title for k in TARGET_KEYWORDS):
            continue
        url = (it.get("URL") or "").strip()
        if not url.startswith("http") or url in done_urls:
            continue
        targets.append({"title": title, "url": url, "date": it.get("DOCRELPUBTIME", "")})
    # 最新优先，但也要尽量覆盖较早的专项解读
    targets.sort(key=lambda x: x["date"], reverse=True)
    logger.info(f"[GOV-T] 命中定向候选 {len(targets)} 条，本次抓取前 {args.max} 条")

    now = datetime.now(CST).isoformat()
    pages, new_recs = 0, 0
    fout = open(out, "a", encoding="utf-8")
    try:
        for t in targets:
            if pages >= args.max:
                break
            try:
                html = fetch(t["url"], args.delay)
                if not html:
                    continue
                title_m = re.search(r"<title>(.*?)</title>", html, re.S)
                page_title = (title_m.group(1).strip() if title_m else t["title"])[:60]
                page_title = re.sub(r"_政策解读_中国政府网$", "", page_title).strip()[:60]
                text = strip_html(extract_main_html(html))
                pairs = extract_qa_pairs(text) or extract_qa_paragraphs(text)
                batch = []
                if pairs:
                    for q, a in pairs:
                        batch.append({
                            "domain": "macro_policy", "question": q, "answer": a,
                            "source_url": t["url"],
                            "source_name": f"中国政府网 · {page_title}",
                            "crawled_at": now,
                        })
                else:
                    body = clean_body(text, page_title)
                    if 100 <= len(body) <= 3000:
                        batch.append({
                            "domain": "macro_policy", "question": page_title,
                            "answer": body, "source_url": t["url"],
                            "source_name": f"中国政府网政策解读 · {t['date']}",
                            "crawled_at": now,
                        })
                if batch:
                    for r in batch:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fout.flush()
                    new_recs += len(batch)
                    pages += 1
                    if pages % 10 == 0:
                        logger.info(f"[GOV-T] 进度 {pages}/{args.max} 页，累计新增 {new_recs} 条")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[GOV-T] 处理 {t['url']} 异常: {e}")
                continue
    finally:
        fout.close()

    total = base_n + new_recs
    logger.info(f"[GOV-T] 完成：新增 {new_recs} 组（{pages} 页），文件累计 {total} 条 -> {out}")


if __name__ == "__main__":
    main()
