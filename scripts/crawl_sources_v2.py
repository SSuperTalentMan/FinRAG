#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/crawl_sources_v2.py — 公开金融问答合规采集（多源扩展版 v2.1）

合规原则：
  - robots.txt 校验：禁止即跳过（pbc.gov.cn 红线不爬）
  - 低频礼貌：请求间隔 >= delay，仅 GET 公开页面；官方公开 API 同样限速
  - 每条记录带 source_url + crawled_at 溯源
  - 仅采集政务公开/投资者教育内容，用于内部知识库研究

来源：
  - csrc: 证监会 监管问答(c100107) + 已知投保局答复/答记者问详情页种子
  - sipf: 12386 热线常见问题（列表 + 已知种子页）
  - sse:  上交所投教站（文章类内容转「标题→正文」知识对）
  - gov:  中国政府网政策库官方搜索 API（单关键词 + 翻页 + 标题过滤）

用法：
    python scripts/crawl_sources_v2.py --max-per-source 60 --delay 1.2
    python scripts/crawl_sources_v2.py --sources csrc,gov
输出：
    data/crawled/{csrc_qa,sipf_qa,sse_qa,gov_qa}.jsonl
"""
import json
import re
import sys
import time
import argparse
import urllib.request
import urllib.parse
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
SSE_BASE = "https://edu.sse.com.cn"
GOV_API = "https://sousuo.www.gov.cn/search-gov/data"

# 证监会已知问答详情页种子（投保局答复 / 答记者问 / 风险警示问答）
CSRC_SEED_URLS = [
    "/csrc/c100028/c1002326/content.shtml",   # 投保局对投资者关注问题的答复
    "/csrc/c100028/c1002496/content.shtml",   # 投保局对网友问题的答复
    "/csrc/c100039/c1362127/content.shtml",   # 投保局负责人就开通12386热线答记者问
    "/hainan/c105461/c1300953/content.shtml", # 非法证券投资咨询风险警示问答(一)
]


def fetch(url: str, delay: float, timeout: int = 25) -> str | None:
    time.sleep(delay)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"抓取失败 {url}: {e}")
        return None


def fetch_json(url: str, delay: float, timeout: int = 25) -> dict | None:
    time.sleep(delay)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                   "Referer": "https://sousuo.www.gov.cn/"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"API 失败 {url[:100]}: {e}")
        return None


def robots_allowed(base_url: str, path: str = "/") -> bool:
    host_part = "/".join(base_url.split("/")[:3])
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(host_part + "/robots.txt")
    try:
        rp.read()
    except Exception:
        logger.info(f"{host_part}/robots.txt 不存在，视为允许（公开页面）")
        return True
    ok = rp.can_fetch(UA, base_url + path)
    logger.info(f"robots 检查 {host_part}: {'允许' if ok else '禁止 —— 跳过'}")
    return ok


def strip_html(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&[a-z]+;", "", text)
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def extract_qa_pairs(text: str) -> list[tuple[str, str]]:
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


def page_title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    return m.group(1).strip() if m else ""


def save(name: str, records: list[dict]):
    out = OUT_DIR / f"{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"{name}: 共 {len(records)} 组 -> {out}")


# ─── 证监会 ──────────────────────────────────────────────────────────────────
def crawl_csrc(max_pages: int, delay: float) -> list[dict]:
    if not robots_allowed(CSRC_BASE, "/csrc/c100107/common_list.shtml"):
        return []
    records, now = [], datetime.now(CST).isoformat()
    detail_links = list(CSRC_SEED_URLS)
    # 监管问答频道列表
    for u in (CSRC_BASE + "/csrc/c100107/common_list.shtml",
              CSRC_BASE + "/csrc/c100107/index_1.shtml"):
        html = fetch(u, delay)
        if not html:
            continue
        for m in re.finditer(r'href="(/csrc/c\d+/c\d+/content\.shtml)"[^>]*>(.*?)</a>', html, re.S):
            href, title = m.group(1), re.sub(r"<[^>]+>|\s+", "", m.group(2))
            if any(k in title for k in ("问答", "答记者问", "答问", "答复")) and href not in detail_links:
                detail_links.append(href)
    logger.info(f"[CSRC] 候选 {len(detail_links)} 篇，抓取前 {max_pages} 篇")
    for href in list(dict.fromkeys(detail_links))[:max_pages]:
        html = fetch(CSRC_BASE + href, delay)
        if not html:
            continue
        pairs = extract_qa_pairs(strip_html(html))
        if not pairs:
            continue
        for q, a in pairs:
            records.append({"domain": "regulatory_qa", "question": q, "answer": a,
                            "source_url": CSRC_BASE + href,
                            "source_name": f"中国证监会 · {page_title(html)[:40]}",
                            "crawled_at": now})
    return records


# ─── 投保基金 12386 ──────────────────────────────────────────────────────────
SIPF_SEED_URLS = [
    "/tzzjy/tjjt/all/2020/04/12926.shtml",  # 香港居民可否购买国内私募基金
    "/tzzjy/tjjt/all/2021/01/13380.shtml",  # 股价波动亏损能否向上市公司索赔
]


def crawl_sipf(max_pages: int, delay: float) -> list[dict]:
    if not robots_allowed(SIPF_BASE, "/tzzjy/tjjt/"):
        return []
    records, now = [], datetime.now(CST).isoformat()
    links = list(SIPF_SEED_URLS)
    html = fetch(SIPF_BASE + "/tzzjy/tjjt/", delay)
    if html:
        for h in re.findall(r'href="(/tzzjy/tjjt/all/\d{4}/\d{2}/\d+\.shtml)"', html):
            if h not in links:
                links.append(h)
    for href in list(dict.fromkeys(links))[:max_pages]:
        html = fetch(SIPF_BASE + href, delay)
        if not html:
            continue
        text = strip_html(html)
        pairs = extract_qa_pairs(text)
        if not pairs:
            am = re.search(r"答\s*[：:]\s*(.+)", text, re.S)
            tm = re.search(r"^(.+?)[\r\n]", text)
            if am and len(am.group(1)) > 30 and tm:
                q = re.sub(r"^.*?[：:]\s*", "", tm.group(1)).strip()
                if len(q) >= 8:
                    pairs = [(q, am.group(1).strip()[:2000])]
        if not pairs:
            continue
        for q, a in pairs:
            records.append({"domain": "investor_protection", "question": q, "answer": a,
                            "source_url": SIPF_BASE + href,
                            "source_name": f"12386热线常见问题 · {page_title(html)[:40]}",
                            "crawled_at": now})
    return records


# ─── 上交所投教站（文章 → 标题/正文知识对）────────────────────────────────────
def crawl_sse(max_pages: int, delay: float) -> list[dict]:
    if not robots_allowed(SSE_BASE, "/"):
        return []
    records, now = [], datetime.now(CST).isoformat()
    links: list[str] = []
    for s in (SSE_BASE + "/", SSE_BASE + "/attention/"):
        html = fetch(s, delay)
        if not html:
            continue
        for m in re.finditer(r'href="(/[^"]+\.shtml)"[^>]*>(.*?)</a>', html, re.S):
            href, title = m.group(1), re.sub(r"<[^>]+>|\s+", "", m.group(2))
            if ("/attention/" in href or "/c/" in href) and "files/" not in href \
                    and len(title) >= 8 and href not in links:
                links.append(href)
    logger.info(f"[SSE] 候选 {len(links)} 页，抓取前 {max_pages} 页")
    for href in list(dict.fromkeys(links))[:max_pages]:
        html = fetch(SSE_BASE + href, delay)
        if not html:
            continue
        text = strip_html(html)
        pairs = extract_qa_pairs(text)
        if not pairs:
            # 文章类：标题 → 正文（正文取有效内容段落）
            body = text
            for cut in ("责任编辑", "免责声明", "版权", "风险提示"):
                idx = body.find(cut)
                if idx > 0:
                    body = body[:idx]
            body = re.sub(r"\s+", " ", body).strip()
            if 80 <= len(body) <= 3000 and len(page_title(html)) >= 8:
                pairs = [(page_title(html), body)]
        if not pairs:
            continue
        for q, a in pairs:
            records.append({"domain": "investor_protection", "question": q, "answer": a,
                            "source_url": SSE_BASE + href,
                            "source_name": f"上交所投教 · {page_title(html)[:40]}",
                            "crawled_at": now})
    return records


# ─── 中国政府网政策库（官方公开搜索 API）──────────────────────────────────────
GOV_KEYWORDS = ["答记者问", "政策问答", "政策解读", "金融", "银行", "债券", "保险", "利率", "外汇"]


def crawl_gov(max_pages: int, delay: float) -> list[dict]:
    if not robots_allowed("https://www.gov.cn", "/zhengce/"):
        return []
    records, now = [], datetime.now(CST).isoformat()
    collected: dict[str, dict] = {}
    for kw in GOV_KEYWORDS:
        for page in range(1, 4):
            q = urllib.parse.quote(kw)
            url = f"{GOV_API}?t=zhengcelibrary_gw&q={q}&timetype=timeqb&sort=score&p={page}&n=10"
            d = fetch_json(url, delay)
            if not d:
                break
            sv = d.get("searchVO") or {}
            items = sv.get("listVO") or []
            if not items:
                break
            for it in items:
                title = (it.get("title") or "").strip()
                if not any(k in title for k in ("答问", "答记者问", "问答", "解读")):
                    continue
                pid = str(it.get("id") or "")
                if pid and pid not in collected:
                    collected[pid] = {"id": pid, "title": title,
                                      "ym": (it.get("pubtimeStr") or "2026.01").replace(".", ""),
                                      "summary": re.sub(r"\s+", " ", it.get("summary") or "").strip()}
        logger.info(f"[GOV] 关键词「{kw}」累计候选 {len(collected)}")
    logger.info(f"[GOV] 标题匹配候选 {len(collected)} 条")
    done = 0
    for it in collected.values():
        if done >= max_pages:
            break
        ok_url, html = None, None
        for pat in (f"/zhengce/zhengceku/{it['ym'][:6]}/content_{it['id']}.htm",
                    f"/zhengce/{it['ym'][:4]}/{it['ym'][4:6]}/content_{it['id']}.htm",
                    f"/zhengce/content_{it['id']}.htm"):
            u = "https://www.gov.cn" + pat
            h = fetch(u, delay)
            if h and "404" not in h[:200] and "Not Found" not in h[:200]:
                ok_url, html = u, h
                break
        if html:
            pairs = extract_qa_pairs(strip_html(html))
            if pairs:
                for q, a in pairs:
                    records.append({"domain": "macro_policy", "question": q, "answer": a,
                                    "source_url": ok_url, "source_name": f"中国政府网 · {it['title'][:40]}",
                                    "crawled_at": now})
                done += 1
                continue
        if it["summary"] and len(it["summary"]) >= 30:
            records.append({"domain": "macro_policy", "question": it["title"],
                            "answer": it["summary"][:2000], "source_url": "https://www.gov.cn/zhengce/",
                            "source_name": "中国政府网政策库（摘要）", "crawled_at": now})
            done += 1
    return records


SOURCES = {"csrc": crawl_csrc, "sipf": crawl_sipf, "sse": crawl_sse, "gov": crawl_gov}


def main():
    ap = argparse.ArgumentParser(description="公开金融问答合规采集（多源）")
    ap.add_argument("--max-per-source", type=int, default=60)
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--sources", type=str, default="csrc,sipf,sse,gov")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("公开金融问答合规采集（多源 v2.1）")
    logger.info("=" * 60)
    for name in [s.strip() for s in args.sources.split(",") if s.strip() in SOURCES]:
        try:
            records = SOURCES[name](args.max_per_source, args.delay)
            save(f"{name}_qa", records)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{name} 采集异常: {e}")


if __name__ == "__main__":
    main()
