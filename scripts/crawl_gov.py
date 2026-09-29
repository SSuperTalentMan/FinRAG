#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/crawl_gov.py — 中国政府网政策解读合规采集

数据链路（全部公开、静态，无需浏览器）：
  1. 列表：https://www.gov.cn/zhengce/jiedu/ZCJD_QZ.json（官方公开 JSON，8 千余条）
  2. 详情：https://www.gov.cn/zhengce/YYYYMM/content_<id>.htm（静态 HTML）
robots.txt：www.gov.cn 允许 /（仅禁 2016 旧路径），低频礼貌抓取（默认 1.2s 间隔）。

特性：
  * 增量落盘 —— 每处理完一页立即 append，进程被杀也不丢数据
  * 断点续爬 —— 重启后自动跳过已抓取的 source_url
  * 金融相关性过滤 —— 强关键词命中 1 次即保留，弱关键词需命中 2 次

输出：data/crawled/gov_qa.jsonl，字段含 source_url / crawled_at 溯源。

用法：
    python scripts/crawl_gov.py --max 300 --delay 1.2
"""
import json
import re
import time
import argparse
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "crawled"
CST = timezone(timedelta(hours=8))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "FinRag-Research/1.0 (+internal RAG research; low-frequency polite crawler)")

LIST_URL = "https://www.gov.cn/zhengce/jiedu/ZCJD_QZ.json"

# 强关键词：命中 1 次即视为金融/经济领域相关
STRONG_KEYWORDS = (
    "金融", "银行", "债券", "证券", "基金", "资本市场", "利率", "汇率", "外汇",
    "货币", "央行", "人民银行", "股市", "理财", "融资", "贷款", "信贷", "支付",
    "数字人民币", "上市公司", "信托", "期货", "资管", "国债", "再融资",
    "保险资金", "商业保险", "保险公司", "金融监管", "货币政策", "财政政策",
    "减税降费", "税费", "财政", "税收", "税制", "储蓄", "存款", "不良资产",
    "小额贷款", "担保", "再保险", "养老金", "社保基金", "住房公积",
)
# 弱关键词：需命中 2 次才保留（单独出现时多指非金融政策）
WEAK_KEYWORDS = ("监管", "投资", "保险", "价格", "补贴", "资金", "消费", "外贸", "企业")


def fetch(url: str, delay: float, timeout: int = 25) -> str | None:
    if delay > 0:
        time.sleep(delay)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"抓取失败 {url}: {e}")
        return None


def robots_ok() -> bool:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url("https://www.gov.cn/robots.txt")
    try:
        rp.read()
    except Exception:
        logger.info("www.gov.cn robots.txt 不可达，视为允许")
        return True
    ok = rp.can_fetch(UA, "https://www.gov.cn/zhengce/jiedu/")
    logger.info(f"www.gov.cn robots: {'允许' if ok else '禁止'}")
    return ok


def strip_html(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&[a-z]+;", "", text)
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


MAIN_MARKERS = ('id="UCAP-CONTENT"', "id='UCAP-CONTENT'", 'class="pages_content"',
                'class="TRS_Editor"', 'id="zoom"')
FOOTER_MARKERS = ("责任编辑", "扫一扫", "相关链接", "Copyright", "京ICP备",
                  "站点地图", "分享到", "打印本页", "关闭窗口", "上一篇", "下一篇")


def extract_main_html(html: str) -> str:
    """
    定位正文容器并按 <div> 配平截取，避免把页头导航（含“中国政府网”等词）
    误当作正文，也避免 clean_body 在导航栏处提前截断。
    找不到容器时回退为整页（去掉 script/style）。
    """
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S | re.I)
    start = -1
    for mk in MAIN_MARKERS:
        start = html.find(mk)
        if start > 0:
            # 移到该标签的结束 '>'
            gt = html.find(">", start)
            start = gt + 1 if gt > 0 else start
            break
    if start <= 0:
        return html
    depth, i, n = 1, start, len(html)
    tag_re = re.compile(r"<(/?)div\b", re.I)
    while i < n and depth > 0:
        m = tag_re.search(html, i)
        if not m:
            break
        depth += -1 if m.group(1) else 1
        i = m.end()
    return html[start:i if depth == 0 else n]


def extract_qa_pairs(text: str) -> list[tuple[str, str]]:
    # 编号（“一、”“1.”等）可缺省，结束位置也允许是无编号的下一个“问：”，
    # 否则一篇答记者问只会抽出第一组。
    num = r"(?:[一二三四五六七八九十\d]+[、.．]?)?"
    pattern = re.compile(
        num + r"\s*问\s*[：:]\s*(.+?)\s*答\s*[：:]\s*(.+?)(?=" + num + r"\s*问\s*[：:]|\Z)",
        re.S,
    )
    pairs = []
    for m in pattern.finditer(text):
        q = re.sub(r"\s+", " ", m.group(1)).strip()
        a = re.sub(r"[ \t]+", " ", m.group(2)).strip()
        if len(q) >= 6 and len(a) >= 20:
            pairs.append((q, a[:2000]))
    return pairs


def extract_qa_paragraphs(text: str) -> list[tuple[str, str]]:
    """
    兜底抽取：gov.cn 大量答记者问采用「一、……？ \n 答：……」格式，
    没有“问：”前缀，正则抽取会全部漏掉。这里按段落配对。
    """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    ans_prefix = re.compile(r"^答\s*[：:]?\s*")
    pairs: list[tuple[str, str]] = []
    i = 0
    n = len(paras)
    while i < n - 1:
        cur = paras[i]
        nxt = paras[i + 1]
        if (cur.endswith("？") or cur.endswith("?")) and ans_prefix.match(nxt):
            q = re.sub(r"^[一二三四五六七八九十\d]+[、.．]?\s*", "", cur).strip()
            chunks = [ans_prefix.sub("", nxt).strip()]
            j = i + 2
            while j < n:
                p = paras[j]
                nxt_is_q = (j + 1 < n and (p.endswith("？") or p.endswith("?"))
                            and ans_prefix.match(paras[j + 1]))
                if nxt_is_q:
                    break
                chunks.append(p)
                j += 1
            a = re.sub(r"[ \t]+", " ", "\n".join(chunks)).strip()
            if len(q) >= 6 and len(a) >= 20:
                pairs.append((q, a[:2000]))
            i = j
        else:
            i += 1
    return pairs


def clean_body(text: str, title: str, min_cut: int = 120) -> str:
    """
    去掉页脚与标题重复。min_cut 保证不会把 120 字以内的正文误截
    （页脚关键词若出现在前 120 字，多半是导航残留，忽略之）。
    """
    body = text
    for cut in FOOTER_MARKERS:
        idx = body.find(cut)
        if idx >= min_cut:
            body = body[:idx]
    body = re.sub(r"\s+", " ", body).strip()
    body = body.replace(title, "", 1).strip()
    return body[:2500]


def is_finance_title(title: str) -> bool:
    if any(k in title for k in STRONG_KEYWORDS):
        return True
    return sum(1 for k in WEAK_KEYWORDS if k in title) >= 2


def load_done(out_path: Path) -> tuple[set[str], int]:
    """读取已有结果，返回 (已抓 URL 集合, 已有记录数)，用于断点续爬。"""
    urls, n = set(), 0
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if o.get("source_url"):
                urls.add(o["source_url"])
    return urls, n


def main():
    ap = argparse.ArgumentParser(description="gov.cn 政策解读合规采集")
    ap.add_argument("--max", type=int, default=300, help="本次最多抓取的详情页数")
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--all-titles", action="store_true",
                    help="关闭金融相关性过滤（默认仅保留金融/经济领域）")
    args = ap.parse_args()

    if not robots_ok():
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "gov_qa.jsonl"

    done_urls, base_n = load_done(out)
    if done_urls:
        logger.info(f"[GOV] 断点续爬：已有 {base_n} 条 / {len(done_urls)} 个页面，自动跳过")

    raw = fetch(LIST_URL, 0)
    if not raw:
        logger.error("无法获取列表 JSON")
        return
    items = json.loads(raw)
    logger.info(f"[GOV] 列表共 {len(items)} 条")

    targets = []
    for it in items:
        title = (it.get("TITLE") or "").strip()
        if not title:
            continue
        if not args.all_titles and not is_finance_title(title):
            continue
        url = (it.get("URL") or "").strip()
        if not url.startswith("http") or url in done_urls:
            continue
        targets.append({"title": title, "url": url,
                        "date": it.get("DOCRELPUBTIME", "")})
    # 最新优先
    targets.sort(key=lambda x: x["date"], reverse=True)
    logger.info(f"[GOV] 金融领域解读候选 {len(targets)} 条（已排除已抓），本次抓取前 {args.max} 条")

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
                        batch.append({"domain": "macro_policy", "question": q, "answer": a,
                                      "source_url": t["url"],
                                      "source_name": f"中国政府网 · {page_title}",
                                      "crawled_at": now})
                else:
                    body = clean_body(text, page_title)
                    if 100 <= len(body) <= 3000:
                        batch.append({"domain": "macro_policy", "question": page_title,
                                      "answer": body, "source_url": t["url"],
                                      "source_name": f"中国政府网政策解读 · {t['date']}",
                                      "crawled_at": now})
                if batch:
                    for r in batch:
                        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fout.flush()
                    new_recs += len(batch)
                    pages += 1
                    if pages % 10 == 0:
                        logger.info(f"[GOV] 进度 {pages}/{args.max} 页，累计新增 {new_recs} 条")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[GOV] 处理 {t['url']} 异常: {e}")
                continue
    finally:
        fout.close()

    total = base_n + new_recs
    logger.info(f"[GOV] 完成：新增 {new_recs} 组（{pages} 页），文件累计 {total} 条 -> {out}")


if __name__ == "__main__":
    main()
