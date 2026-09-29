#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/crawl_public_faq.py — 公开金融监管问答合规采集

合规原则（红线）：
  1. 严格尊重 robots.txt：对每个站点先用 urllib.robotparser 校验本 UA 是否允许，
     禁止爬取的站点直接跳过（如 pbc.gov.cn 明确 Disallow: / —— 绝不爬取）。
  2. 低频礼貌抓取：每次请求间隔 >= 2 秒，只做 GET，只访问公开页面。
  3. 只采集"政务公开/投资者教育"性质的问答内容，用于内部知识库研究，
     不转售、不对外分发原文。
  4. 每条数据记录 source_url 与抓取时间，保证数据可溯源。

目标站点（均已核查 robots.txt）：
  - 中国证监会「监管问答」栏目  https://www.csrc.gov.cn/csrc/c100107/common_list.shtml
    （robots.txt 不存在，政务公开信息）
  - 中国证券投资者保护基金 12386 热线常见问题  https://www.sipf.com.cn
    （robots.txt 不存在，公益投资者教育内容）

用法：
    python scripts/crawl_public_faq.py                # 默认每源最多 30 篇
    python scripts/crawl_public_faq.py --max-per-source 60
    python scripts/crawl_public_faq.py --delay 3      # 提高抓取间隔
输出：
    data/crawled/csrc_qa.jsonl  /  sipf_qa.jsonl
    字段：{domain, question, answer, source_url, crawled_at, source_name}
"""
import io
import json
import re
import sys
import time
import argparse
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "crawled"
CST = timezone(timedelta(hours=8))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "FinRag-Research/1.0 (+internal RAG research; low-frequency polite crawler)")

CSRC_BASE = "https://www.csrc.gov.cn"
SIPF_BASE = "https://www.sipf.com.cn"

CSRC_QA_CHANNEL = "/csrc/c100107/common_list.shtml"   # 监管问答
CSRC_TZBH_CHANNEL = "/csrc/c100028/c1002326/content.shtml"  # 投保局答复(示例详情)


def fetch(url: str, delay: float, timeout: int = 25) -> str | None:
    """礼貌抓取单页，失败返回 None。"""
    time.sleep(delay)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"抓取失败 {url}: {e}")
        return None


def robots_allowed(base_url: str, path: str = "/") -> bool:
    """检查 robots.txt 是否允许本 UA 抓取（无 robots.txt 视为允许）。"""
    host_part = "/".join(base_url.split("/")[:3])
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(host_part + "/robots.txt")
    try:
        rp.read()
    except Exception:
        logger.info(f"{host_part}/robots.txt 不存在或不可达，视为允许（公开页面）")
        return True
    ok = rp.can_fetch(UA, base_url + path)
    logger.info(f"robots.txt 检查 {host_part}: {'允许' if ok else '禁止 —— 跳过该站'}")
    return ok


def strip_html(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&[a-z]+;", "", text)
    lines = [l.strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


def extract_qa_pairs(text: str) -> list[tuple[str, str]]:
    """从正文提取 问/答 对（兼容 问：/答： 与 问: /答: 格式）。"""
    pattern = re.compile(
        r"([一二三四五六七八九十\d]+、)?\s*问\s*[：:]\s*(.+?)\s*答\s*[：:]\s*(.+?)(?=[一二三四五六七八九十\d]+、\s*问\s*[：:]|\Z)",
        re.S,
    )
    pairs = []
    for m in pattern.finditer(text):
        q = re.sub(r"\s+", " ", m.group(2)).strip()
        a = re.sub(r"[ \t]+", " ", m.group(3)).strip()
        if len(q) >= 6 and len(a) >= 20:
            pairs.append((q, a[:2000]))
    return pairs


def crawl_csrc(max_pages: int, delay: float) -> list[dict]:
    """爬取证监会监管问答/答记者问类详情页，提取问答对。"""
    if not robots_allowed(CSRC_BASE, CSRC_QA_CHANNEL):
        return []
    records: list[dict] = []
    # 列表页翻页（csrc 列表页为 index_1.shtml 形式）
    list_urls = [CSRC_BASE + CSRC_QA_CHANNEL]
    for i in range(1, 6):
        list_urls.append(f"{CSRC_BASE}/csrc/c100107/index_{i}.shtml")

    detail_links: list[str] = []
    for lu in list_urls:
        if len(detail_links) >= max_pages:
            break
        html = fetch(lu, delay)
        if not html:
            continue
        for m in re.finditer(r'href="(/csrc/c\d+/c\d+/content\.shtml)"[^>]*>(.*?)</a>', html, re.S):
            href, title = m.group(1), re.sub(r"<[^>]+>|\s+", "", m.group(2))
            if any(k in title for k in ("问答", "答记者问", "答问", "答复")) and href not in detail_links:
                detail_links.append(href)

    logger.info(f"[CSRC] 发现 {len(detail_links)} 篇问答类详情页，最多抓取 {max_pages} 篇")
    now = datetime.now(CST).isoformat()
    for href in detail_links[:max_pages]:
        html = fetch(CSRC_BASE + href, delay)
        if not html:
            continue
        text = strip_html(html)
        pairs = extract_qa_pairs(text)
        if not pairs:
            continue
        title_m = re.search(r"<title>(.*?)</title>", html, re.S)
        page_title = title_m.group(1).strip() if title_m else ""
        for q, a in pairs:
            records.append({
                "domain": "regulatory_qa",
                "question": q,
                "answer": a,
                "source_url": CSRC_BASE + href,
                "source_name": f"中国证监会 · {page_title[:40]}",
                "crawled_at": now,
            })
        logger.info(f"[CSRC] {href} 提取 {len(pairs)} 组问答")
    return records


def crawl_sipf(max_pages: int, delay: float) -> list[dict]:
    """爬取证券投资者保护基金 12386 常见问题。"""
    if not robots_allowed(SIPF_BASE, "/tzzjy/tjjt/"):
        return []
    records: list[dict] = []
    # 已知的 12386 热线常见问题详情页（问答格式规整），由栏目列表发现
    seed_list = SIPF_BASE + "/tzzjy/tjjt/"
    html = fetch(seed_list, delay)
    if not html:
        return records
    links = []
    for m in re.finditer(r'href="(/tzzjy/tjjt/all/\d{4}/\d{2}/\d+\.shtml)"[^>]*>(.*?)</a>', html, re.S):
        href, title = m.group(1), re.sub(r"<[^>]+>|\s+", "", m.group(2))
        if ("常见问题" in title or "热线" in title) and href not in links:
            links.append(href)
    # 列表页只展示近期案例时，回退到搜索引擎发现的已知 FAQ 种子页
    if not links:
        links = [
            "/tzzjy/tjjt/all/2020/04/12926.shtml",  # 香港地区居民可否购买国内私募基金
            "/tzzjy/tjjt/all/2021/01/13380.shtml",  # 股价波动亏损能否向上市公司索赔
        ]
    links = list(dict.fromkeys(links))
    logger.info(f"[SIPF] 候选 {len(links)} 个 FAQ 页面")
    now = datetime.now(CST).isoformat()
    for href in links[:max_pages]:
        html = fetch(SIPF_BASE + href, delay)
        if not html:
            continue
        text = strip_html(html)
        pairs = extract_qa_pairs(text)
        if not pairs:
            # 单问单答页：标题即问题，正文含"答："
            am = re.search(r"答\s*[：:]\s*(.+)", text, re.S)
            tm = re.search(r"^(.+?)[\r\n]", text)
            if am and len(am.group(1)) > 30:
                q = re.sub(r"^.*?[：:]\s*", "", tm.group(1)).strip() if tm else ""
                a = am.group(1).strip()[:2000]
                if len(q) >= 8:
                    pairs = [(q, a)]
        if not pairs:
            continue
        title_m = re.search(r"<title>(.*?)</title>", html, re.S)
        page_title = title_m.group(1).strip() if title_m else ""
        for q, a in pairs:
            records.append({
                "domain": "investor_protection",
                "question": q,
                "answer": a,
                "source_url": SIPF_BASE + href,
                "source_name": f"12386热线常见问题 · {page_title[:40]}",
                "crawled_at": now,
            })
        logger.info(f"[SIPF] {href} 提取 {len(pairs)} 组问答")
    return records


def main():
    ap = argparse.ArgumentParser(description="公开金融监管问答合规采集")
    ap.add_argument("--max-per-source", type=int, default=30, help="每站点最多抓取的详情页数")
    ap.add_argument("--delay", type=float, default=2.0, help="请求间隔秒数（礼貌爬取）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("公开金融问答合规采集启动（尊重 robots.txt / 低频 / 可溯源）")
    logger.info("=" * 60)

    for name, fn in [("csrc_qa", crawl_csrc), ("sipf_qa", crawl_sipf)]:
        try:
            records = fn(args.max_per_source, args.delay)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{name} 采集异常: {e}")
            continue
        out = OUT_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"{name}: 共 {len(records)} 组问答 -> {out}")


if __name__ == "__main__":
    main()
