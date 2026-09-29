#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/crawl_dividend.py — 定向补充「上市公司现金分红」权威内容（合规爬取）

目标：填补 Q4「上市公司分红有哪些规定」的检索覆盖缺口。gov.cn 政策解读库
缺少专门的「上市公司现金分红」文档，故从以下 robots 校验通过的权威静态站点补充：
  - 上交所投教（edu.sse.com.cn）：现金分红指引 / 监管指引第3号 解读
  - 摩根士丹利证券投教：分红定义 / 形式 / 除权除息 / 强制分红措施（30% 规则）20 问
  - 人民网 / 央视网：2024 新“国九条”分红 ST 硬约束 + 一年多次分红
  - 百度百科：A股现金分红「政策规定」汇总（监管指引第3号/第10号、ST 量化指标）

合规：robots.txt 校验（禁止即跳过）；仅 GET；间隔 >= delay；带 UA + 业务说明；
      每条记录带 source_url + crawled_at 溯源。输出追加到 data/crawled/dividend_qa.jsonl
      （按 source_url+question 去重，可重复运行断点续爬）。

领域：所有记录固定 category=stock_market（对齐 Q4 意图路由，确保 Milvus 严格过滤可见）。

用法：
    python scripts/crawl_dividend.py --test            # 仅打印抽取结果不落盘
    python scripts/crawl_dividend.py --delay 1.2       # 正式爬取并追加落盘
"""
import json
import re
import sys
import time
import argparse
import ssl
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CRAWLED_DIR = PROJECT_ROOT / "data" / "crawled"
OUT = CRAWLED_DIR / "dividend_qa.jsonl"
CST = timezone(timedelta(hours=8))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "FinRag-Research/1.0 (+internal RAG research; low-frequency polite crawler)")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# (url, source_name) —— 均为公开投教/政策解读内容（robots 校验通过且可静态抽取）
# 注：人民网/央视网新闻页正文隔离困难（抽取 0 条），百度百科 robots 禁止，已剔除；
#     csrc.gov.cn 与 morganstanley 此前偶发代理 502，运行时自动跳过不影响其它来源。
TARGETS = [
    ("https://edu.sse.com.cn/best/article/mcontent/c/4731127.shtml",
     "上交所投教 · 现金分红：稳稳的幸福（第三期）"),
    ("https://www.morganstanleysecurities.com.cn/investor/education-dividendpolicy.html",
     "摩根士丹利证券投资者教育 · 什么是分红（20 问）"),
    # 2024 新“国九条”分红 ST 硬约束（官方交易所答记者问，最权威且静态可抽）
    ("https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20240412_10753148.shtml",
     "上交所 · 就股票上市规则等修订答记者问（现金分红新变化）"),
    ("https://www.szse.cn/aboutus/trends/conference/t20240412_606844.html",
     "深交所 · 2024年4月12日新闻发布会（现金分红新变化）"),
    ("https://www.szse.cn/aboutus/trends/conference/t20240430_607071.html",
     "深交所 · 2024年4月30日新闻发布会（分红不达标ST具体考虑）"),
]

BOILER = ["免责声明", "责任编辑", "编辑：", "本文来源", "来源：", "点击", "扫一扫",
          "关注我们", "相关阅读", "上一篇", "下一篇", "分享到", "展开全文", "参考资料",
          "責任編輯", "分享", "相關", "免責聲明", "資料來源", "關鍵字", "標籤",
          "打開", "首頁", "经济·科技", "財經頻道", "财经频道"]

# 只保留与「现金分红」高度相关的记录，避免混入混合问答里的无关条目
RELEVANT = ("分红", "现金分红", "派息", "除权", "除息", "股息", "红利", "派现", "现金红利",
            "現金分紅", "分紅", "股息", "紅利")
# QA 对：要求「问题」本身即关于分红（避免答记者问里 回购/债券 等相邻话题误入）
QA_RELEVANT = ("分红", "现金分红", "派息", "股息", "红利", "除权", "除息", "現金分紅", "分紅")


def is_relevant_qa(q: str, a: str) -> bool:
    return any(k in q for k in QA_RELEVANT)


def is_relevant_article(q: str, a: str) -> bool:
    return any(k in (q + a) for k in RELEVANT)


# ── 网络 ──────────────────────────────────────────────────────────────────────
def robots_allowed(url: str) -> bool:
    from urllib.parse import urlparse
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    rp = RobotFileParser()
    rp.set_url(base + "/robots.txt")
    try:
        rp.read()
    except Exception:
        return True  # 无 robots 视为允许（公开页面）
    ok = rp.can_fetch(UA, url)
    print(f"  robots {base}: {'允许' if ok else '禁止 → 跳过'}")
    return ok


def fetch(url: str, delay: float, timeout: int = 30) -> str | None:
    time.sleep(delay)
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        print(f"  抓取失败 {url}: {e}")
        return None


# ── 文本清洗 ────────────────────────────────────────────────────────────────────
def strip_html(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>|</div>|</tr>|</li>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&nbsp;?|&amp;?", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cut_body(text: str, title: str = "") -> str:
    """去掉页头导航与页尾免责声明等噪声，保留正文。"""
    # 上交所投教：从【编者按】开始
    for mk in ("【编者按】", "编者按：", "编者按:"):
        i = text.find(mk)
        if i > 0:
            text = text[i + len(mk):]
            break
    else:
        # 其它：去掉页头导航——标题通常在导航栏与正文 h1 各出现一次，
        # 从标题第二次出现处开始即为正文。
        if title:
            p = text.find(title)
            if p > 0:
                p2 = text.find(title, p + len(title))
                if p2 > 0:
                    text = text[p2 + len(title):]
        # 仍找不到则从「日期+来源」行之后开始
        m = re.search(r"\d{4}[-年]\d{1,2}[-月]\d{1,2}\s*日?\s*来源[：:].*?(?=\S)", text)
        if m and len(text) - m.end() > 200:
            text = text[m.end():]
    # 去掉页尾 boilerplate（简体 + 繁体）
    for cut in BOILER:
        idx = text.find(cut)
        if 0 < idx < len(text) - 40:
            text = text[:idx]
    return text.strip()


# ── 问答抽取 ──────────────────────────────────────────────────────────────────
def extract_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    # 1) Qn. ...? 答： （摩根士丹利 20 问）
    for m in re.finditer(
        r"Q\s*(\d+)[.、)]\s*([^？\n]*?[?？])\s*答\s*[：:]\s*(.+?)(?=Q\s*\d+[.、)]\s*[^？\n]*?[?？]|\Z)",
        text, re.S,
    ):
        q = clean(m.group(2)); a = clean(m.group(3))
        if len(q) >= 4 and len(a) >= 20:
            pairs.append((q, a[:2000]))
    if pairs:
        return pairs
    # 2) 问：...答：
    for m in re.finditer(r"问\s*[：:]\s*(.+?)\s*答\s*[：:]\s*(.+?)(?=问\s*[：:]|\Z)", text, re.S):
        q = clean(m.group(1)); a = clean(m.group(2))
        if len(q) >= 4 and len(a) >= 20:
            pairs.append((q, a[:2000]))
    if pairs:
        return pairs
    # 3) 一、...？ ... 二、 （上交所现金分红：稳稳的幸福 风格）
    for m in re.finditer(
        r"\n([一二三四五六七八九十]+)、([^\n]+?[?？][^\n]*)\n([\s\S]*?)(?=\n[一二三四五六七八九十]+、|\Z)",
        text,
    ):
        q = clean(m.group(2))
        q = re.sub(r"[（(][^（）()]*[）)]$", "", q).strip()  # 去掉“（武汉投资者…来信询问）”
        a = clean(m.group(3))
        if len(q) >= 6 and len(a) >= 30:
            pairs.append((q, a[:2000]))
    # 4) 答记者问：抓取「N、<问>。 答：<答>」整段（上交所/深交所官网 新国九条 分红 ST 规则）
    #    答中若含「一是对…/二是…」等子条目，会被一并纳入，直到下一个同样带「答：」的提问标题为止。
    for m in re.finditer(
        r"[（(]?([一二三四五六七八九十\d]+)[）)、、]\s*([^。\n]{2,60}[。]?)\s*答\s*[：:]\s*(.+?)"
        r"(?=[（(]?[一二三四五六七八九十\d]+[）)、、]\s*[^。\n]{2,60}[。]?\s*答\s*[：:]|\Z)",
        text, re.S,
    ):
        q = clean(m.group(2))
        a = clean(re.sub(r"^\s*答\s*[：:]\s*", "", m.group(3)))
        if len(q) >= 4 and len(a) >= 40:
            pairs.append((q, a[:2000]))
    return pairs


def article_record(title: str, text: str, source_name: str, source_url: str, now: str) -> dict | None:
    body = cut_body(text, title)
    body = re.sub(r"\s+", " ", body).strip()
    body = body[:1800]
    if len(body) < 80:
        return None
    # 让问题文本带“分红/上市公司”以强化检索与领域命中
    if any(t in title for t in ("分红", "现金分红", "上市公司", "股票", "A股")):
        q = title
    else:
        q = f"{title}（上市公司现金分红相关规定）"
    return {
        "category": "stock_market",
        "question": q,
        "answer": body,
        "source_url": source_url,
        "source_name": source_name,
        "crawled_at": now,
    }


def page_title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if m:
        return clean(m.group(1).split("|")[0]).strip()
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    if m:
        return clean(re.sub(r"<[^>]+>", "", m.group(1)))
    return ""


# ── 主流程 ──────────────────────────────────────────────────────────────────────
def crawl_one(url: str, source_name: str, delay: float) -> list[dict]:
    if not robots_allowed(url):
        return []
    html = fetch(url, delay)
    if not html:
        return []
    text = strip_html(html)
    title = page_title(html)
    now = datetime.now(CST).isoformat()
    pairs = extract_pairs(text)
    recs: list[dict] = []
    seen_q: set[str] = set()
    if pairs:
        for q, a in pairs:
            if q in seen_q:               # 页内去重（防止正则重叠多抓）
                continue
            if not is_relevant_qa(q, a):   # 混合问答（如上交所每月问答）只留分红相关
                continue
            seen_q.add(q)
            recs.append({
                "category": "stock_market",
                "question": q,
                "answer": a,
                "source_url": url,
                "source_name": source_name,
                "crawled_at": now,
            })
    else:
        art = article_record(title, text, source_name, url, now)
        if art and is_relevant_article(art["question"], art["answer"]):
            recs.append(art)
    print(f"  {source_name}: 抽取 {len(recs)} 条")
    return recs


def load_done() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((o.get("source_url", ""), o.get("question", "")))
    return done


def main():
    ap = argparse.ArgumentParser(description="定向补充上市公司现金分红内容（合规爬取）")
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--test", action="store_true", help="仅打印抽取结果，不落盘")
    args = ap.parse_args()
    CRAWLED_DIR.mkdir(parents=True, exist_ok=True)
    done = set() if args.test else load_done()

    all_recs: list[dict] = []
    for url, name in TARGETS:
        print(f"[抓取] {url}")
        recs = crawl_one(url, name, args.delay)
        for r in recs:
            key = (r["source_url"], r["question"])
            if key in done:
                print(f"    跳过已存在: {r['question'][:30]}")
                continue
            done.add(key)
            all_recs.append(r)

    print(f"\n合计新增 {len(all_recs)} 条")
    if args.test:
        for r in all_recs:
            print(f"\n● [{r['source_name']}]")
            print(f"  Q: {r['question']}")
            print(f"  A: {r['answer'][:200]}...")
        return
    # 落盘（追加）
    with open(OUT, "a", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已追加至 {OUT}")


if __name__ == "__main__":
    main()
