#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/crawl_kechuang.py — 定向补充「科创板注册制改革方向」权威内容（合规爬取）

目标：根治 Q3「科创板注册制改革方向」的检索弱命中（端到端复跑 Top1 仅 0.061，虽内容相关
但分数偏低——库内多为泛资本市场投融资综合改革文档，缺「科创板注册制」专项条目）。
根因同 Q2/Q4：缺针对性语料。本脚本补权威专项内容。

补充源（robots 校验通过 + 静态可抽 + 权威）：
  - 中国政府网·部门动态（证监会）《关于深化科创板改革 服务科技创新和新质生产力发展的
    八条措施》（科创板八条，2024-06）：硬科技定位 / 发行承销试点 / 股债融资 / 并购重组 /
    股权激励 / 交易机制 / 全链条监管 / 市场生态 —— 正对「改革方向」。
  - 证监会新闻发言人就《关于在科创板设置科创成长层…》答记者问（2025-06）：改革背景 /
    总体思路 / 科创成长层考虑 / 投资者保护 / 6 项举措 —— 最新改革方向（问：答： 结构）。
    （csrc.gov.cn 此前偶发代理 502，运行时若失败自动跳过，gov.cn 单源已足够。）

领域：所有记录固定 category=financial_markets（对齐 Q3 意图路由）。

合规：robots.txt 校验；仅 GET；间隔 >= delay；带 UA + 溯源。输出追加到
      data/crawled/kechuang_qa.jsonl（按 source_url+question 去重，可重复运行断点续爬）。

用法：
    python scripts/crawl_kechuang.py --test            # 仅打印抽取结果不落盘
    python scripts/crawl_kechuang.py --delay 1.2       # 正式爬取并追加落盘
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
OUT = CRAWLED_DIR / "kechuang_qa.jsonl"
CST = timezone(timedelta(hours=8))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "FinRag-Research/1.0 (+internal RAG research; low-frequency polite crawler)")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# (url, source_name)
TARGETS = [
    ("https://www.gov.cn/lianbo/bumen/202406/content_6958236.htm",
     "中国政府网 · 证监会《科创板改革八条措施》"),
    ("https://www.csrc.gov.cn/csrc/c100028/c7565135/content.shtml",
     "中国证监会 · 科创成长层意见答记者问"),
]

# 仅页尾免责/版权类噪声（不可包含「来源：」「打印」「字号」等正文头部常见词，
# 否则会把正文一起裁掉）。
BOILER = ["免责声明", "责任编辑", "本文来源", "相关阅读", "上一篇", "下一篇",
          "分享到", "展开全文", "参考资料", "責任編輯", "免責聲明", "資料來源",
          "關鍵字", "標籤", "网站地图", "联系我们", "京公网安备", "版权所有",
          "All Rights Reserved", "ICP备", "主办单位", "政府网站年度报表"]

RELEVANT = ("科创板", "注册制", "资本市场", "上市", "硬科技", "科創板", "註冊制", "資本")
QA_RELEVANT = ("科创板", "注册制", "资本", "上市", "改革", "科創板", "註冊制")


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
        return True
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
    html = re.sub(r"</p>|</div>|</tr>|</li>|</h[1-6]>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&nbsp;?|&amp;?", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# 正文起始锚点：优先从正文开篇处截断，绕过页头 nav / og:title 副本 / 页脚标题重复
START_MARKERS = [
    "各派出机构，各交易所",          # 科创板八条 正文开篇（salutation 后接 preamble）
    "6月18日，中国证监会发布实施",    # 科创成长层 答记者问 正文开篇
    "问：",                          # 答记者问 兜底
    "一、",                          # 编号措施 兜底
]


def cut_body(text: str, title: str = "") -> str:
    """去掉页头导航与页尾免责声明等噪声，保留正文。"""
    for mk in START_MARKERS:
        p = text.find(mk)
        if p > 0:
            text = text[p:]
            break
    # 去掉页尾 boilerplate（仅页尾免责/版权类，不含正文头部常见词）
    for cut in BOILER:
        idx = text.find(cut)
        if 0 < idx < len(text) - 40:
            text = text[:idx]
    return text.strip()


# ── 问答抽取 ──────────────────────────────────────────────────────────────────
def extract_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    # 1) 问：...答： （证监会 科创成长层 答记者问：一、问：...答：... 二、问：...）
    for m in re.finditer(r"问\s*[：:]\s*(.+?)\s*答\s*[：:]\s*(.+?)(?=问\s*[：:]|\Z)", text, re.S):
        q = clean(m.group(1)); a = clean(m.group(2))
        if len(q) >= 4 and len(a) >= 20:
            pairs.append((q, a[:2000]))
    if pairs:
        return pairs
    # 2) 编号措施「N、<标题>。<正文>」（证监会 科创板八条：一、强化科创板硬科技定位。...）
    for m in re.finditer(
        r"\n([一二三四五六七八九十]+)、([^。\n]{2,40}。)\s*([\s\S]*?)"
        r"(?=\n[一二三四五六七八九十]+、|\Z)",
        text,
    ):
        q = clean(m.group(2))
        a = clean(m.group(3))
        if len(q) >= 4 and len(a) >= 30:
            pairs.append((q, a[:2000]))
    if pairs:
        return pairs
    # 3) 行式短问句(含？) + 后续行即答案（兜底）
    raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lines: list[str] = []
    for ln in raw_lines:
        if lines and lines[-1][-1:] not in "。！？…：；":
            lines[-1] = lines[-1] + ln
        else:
            lines.append(ln)
    cur_q = None
    cur_a: list[str] = []
    qline = re.compile(r"^[\u4e00-\u9fa5A-Za-z，、\s]{2,20}[?？]$")
    for ln in lines:
        if qline.match(ln):
            if cur_q and cur_a:
                a = clean(" ".join(cur_a))
                if len(a) >= 30:
                    pairs.append((cur_q, a[:2000]))
            cur_q = ln
            cur_a = []
        elif cur_q is not None:
            cur_a.append(ln)
    if cur_q and cur_a:
        a = clean(" ".join(cur_a))
        if len(a) >= 30:
            pairs.append((cur_q, a[:2000]))
    return pairs


def article_record(title: str, text: str, source_name: str, source_url: str, now: str) -> dict | None:
    body = cut_body(text, title)
    body = re.sub(r"\s+", " ", body).strip()
    body = body[:1800]
    if len(body) < 80:
        return None
    q = f"{title}（科创板注册制改革相关内容）" if "科创" in title or "注册" in title else title
    return {
        "category": "financial_markets",
        "question": q,
        "answer": body,
        "source_url": source_url,
        "source_name": source_name,
        "crawled_at": now,
    }


def page_title(html: str) -> str:
    # 优先取 <h1>（多为干净正文标题），再回退 <title>
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    if m:
        t = clean(re.sub(r"<[^>]+>", "", m.group(1)))
        # 跳过页脚/无障碍等伪 h1（如「政府网站年度报表」）
        if t and len(t) >= 8 and "年度报表" not in t and "政府网站" not in t:
            return t
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if m:
        t = clean(m.group(1))
        # 去掉站点后缀（_中国证券监督管理委员会 / _部门动态_中国政府网 / _政策解读_中国政府网 等）
        for suf in ("_中国证券监督管理委员会", "_部门动态_中国政府网",
                   "_政策解读_中国政府网", "_中国政府网"):
            if t.endswith(suf):
                t = t[: -len(suf)]
        return t.strip()
    return ""


# ── 主流程 ──────────────────────────────────────────────────────────────────────
def crawl_one(url: str, source_name: str, delay: float) -> list[dict]:
    if not robots_allowed(url):
        return []
    html = fetch(url, delay)
    if not html:
        return []
    raw = strip_html(html)
    title = page_title(html)
    text = cut_body(raw, title)
    now = datetime.now(CST).isoformat()
    pairs = extract_pairs(text)
    recs: list[dict] = []
    seen_q: set[str] = set()
    if pairs:
        for q, a in pairs:
            if q in seen_q:
                continue
            if not is_relevant_article(q, a):
                continue
            # 问题补全主题词：本页问句多为通用措辞（总体思路/投资者保护），
            # 补「科创板」前缀使其自描述，强化对「科创板注册制改革方向」的命中。
            if "科创板" not in q and "注册" not in q:
                q = "科创板" + q
            seen_q.add(q)
            recs.append({
                "category": "financial_markets",
                "question": q,
                "answer": a,
                "source_url": url,
                "source_name": source_name,
                "crawled_at": now,
            })
    else:
        art = article_record(title, raw, source_name, url, now)
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
    ap = argparse.ArgumentParser(description="定向补充科创板注册制改革内容（合规爬取）")
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
    with open(OUT, "a", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已追加至 {OUT}")


if __name__ == "__main__":
    main()
