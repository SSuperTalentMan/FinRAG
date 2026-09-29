#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/crawl_pension.py — 定向补充「个人养老金怎么参加」权威内容（合规爬取）

目标：填补 Q2「个人养老金怎么参加」的检索覆盖缺口。端到端重跑（2026-09-01）显示该问
Top1 仅 0.035，#2-5 全是公积金/个税/翻译 FAQ 答非所问——根因与 Q4 当初相同：库内
个人养老金文档多为「领取情形/基本养老金≠个人养老金/全面实施」，缺「参加流程
（开户→缴费→选购→领取）」专项条目；"怎么参加"问法匹配不到。

补充源（robots 校验通过 + 静态可抽 + 权威）：
  - 中国政府网·部门动态（人社部微信公号）：个人养老金怎么开户缴费?资金账户可变更银行吗?
    直接覆盖 谁能参加 / 如何开户缴费 / 可抵扣多少个税 / 如何领取 —— 正对 Q2。
  - 玉林市人社局：个人养老金热门知识问答（01-05 问：答： 结构，强化 参加流程/领取）

领域：所有记录固定 category=personal_finance（对齐 Q2 意图路由，确保 Milvus 严格过滤可见）。

合规：robots.txt 校验；仅 GET；间隔 >= delay；带 UA + 溯源。输出追加到
      data/crawled/pension_qa.jsonl（按 source_url+question 去重，可重复运行断点续爬）。

用法：
    python scripts/crawl_pension.py --test            # 仅打印抽取结果不落盘
    python scripts/crawl_pension.py --delay 1.2       # 正式爬取并追加落盘
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
OUT = CRAWLED_DIR / "pension_qa.jsonl"
CST = timezone(timedelta(hours=8))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "FinRag-Research/1.0 (+internal RAG research; low-frequency polite crawler)")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# (url, source_name) —— 均为公开官方投教/政策解读（robots 校验通过且可静态抽取）
TARGETS = [
    ("https://www.gov.cn/lianbo/bumen/202311/content_6917464.htm",
     "中国政府网 · 个人养老金怎么开户缴费（人社部）"),
    ("http://rsj.yulin.gov.cn/bmzl/shbz/rdbswd/t19778966.shtml",
     "玉林市人力资源和社会保障局 · 个人养老金热门知识问答"),
]

BOILER = ["免责声明", "责任编辑", "编辑：", "本文来源", "来源：", "点击", "扫一扫",
          "关注我们", "相关阅读", "上一篇", "下一篇", "分享到", "展开全文", "参考资料",
          "責任編輯", "分享", "相關", "免責聲明", "資料來源", "關鍵字", "標籤",
          "打開", "首頁", "返回首页", "返回顶部", "网站地图", "联系我们", "网站标识码",
          "京公网安备", "版权所有", "All Rights Reserved", "ICP备", "主办单位"]

# 只保留与「个人养老金」高度相关的记录
RELEVANT = ("个人养老金", "养老金", "养老金融", "個人養老金", "養老金")
QA_RELEVANT = ("个人养老金", "养老金", "养老", "個人養老金", "養老")


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
    html = re.sub(r"</p>|</div>|</tr>|</li>|</h[1-6]>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"&nbsp;?|&amp;?", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cut_body(text: str, title: str = "") -> str:
    """去掉页头导航与页尾免责声明等噪声，保留正文。"""
    if title:
        p = text.find(title)
        if p > 0:
            p2 = text.find(title, p + len(title))
            if p2 > 0:
                text = text[p2 + len(title):]
    # 中国政府网 人社部 开户缴费问答：正文前有一段引导语（含「您参加个人养老金了吗?」
    # 等修辞设问），会污染抽取；从「这篇为您解答」起截断，保留真正的问答标题。
    i = text.find("这篇为您解答")
    if i > 0:
        text = text[i:]
    # 去掉页尾 boilerplate（简体 + 繁体）
    for cut in BOILER:
        idx = text.find(cut)
        if 0 < idx < len(text) - 40:
            text = text[:idx]
    return text.strip()


# ── 问答抽取 ──────────────────────────────────────────────────────────────────
def extract_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    # 1) 问：...答： （玉林市人社局 01-05 结构）
    for m in re.finditer(r"问\s*[：:]\s*(.+?)\s*答\s*[：:]\s*(.+?)(?=问\s*[：:]|\Z)", text, re.S):
        q = clean(m.group(1)); a = clean(m.group(2))
        if len(q) >= 4 and len(a) >= 20:
            pairs.append((q, a[:2000]))
    if pairs:
        return pairs
    # 2) 行式「短问句(含？) + 后续行即答案」（中国政府网 人社部 开户缴费问答：
    #    「谁能参加个人养老金？」「如何开户缴费？」「可以抵扣多少个税？」「如何领取？」
    #    每个问题独占一行，答案为其后的连续行，直到下一个问题行。避免相邻问句互相吞噬字符。
    #    预处理：若一行不以句末标点（。！？…）结尾，则与下一行合并，修复被换行切断的问题
    #    （如「可购买的金融产品」+「及相应收益哪里查？」应合并为一个完整问句）。
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
    if pairs:
        return pairs
    # 3) 一、...？ ... （投教文章风格兜底）
    for m in re.finditer(
        r"\n([一二三四五六七八九十]+)、([^\n]+?[?？][^\n]*)\n([\s\S]*?)(?=\n[一二三四五六七八九十]+、|\Z)",
        text,
    ):
        q = clean(m.group(2))
        q = re.sub(r"[（(][^（）()]*[）)]$", "", q).strip()
        a = clean(m.group(3))
        if len(q) >= 6 and len(a) >= 30:
            pairs.append((q, a[:2000]))
    return pairs


def article_record(title: str, text: str, source_name: str, source_url: str, now: str) -> dict | None:
    body = cut_body(text, title)
    body = re.sub(r"\s+", " ", body).strip()
    body = body[:1800]
    if len(body) < 80:
        return None
    # 让问题文本带“个人养老金”以强化检索与领域命中
    q = f"{title}（个人养老金参加、开户缴费与领取）" if "养老" in title else title
    return {
        "category": "personal_finance",
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
    raw = strip_html(html)
    title = page_title(html)
    text = cut_body(raw, title)        # 先去导航/引导语/页尾，再抽取，避免修辞设问污染
    now = datetime.now(CST).isoformat()
    pairs = extract_pairs(text)
    recs: list[dict] = []
    seen_q: set[str] = set()
    if pairs:
        for q, a in pairs:
            if q in seen_q:
                continue
            # 相关性：整页为个人养老金专题，放宽到「问+答」含养老关键词即可
            # （本页问题多为「如何开户缴费？」等通用措辞，关键词在答案中）。
            if not is_relevant_article(q, a):
                continue
            # 问题补全主题词：本页问句多为通用措辞（如何开户缴费/如何领取），
            # 补「个人养老金」前缀使其自描述，强化对「个人养老金怎么参加」的稀疏/语义命中。
            if "个人养老金" not in q:
                q = "个人养老金" + q
            seen_q.add(q)
            recs.append({
                "category": "personal_finance",
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
    ap = argparse.ArgumentParser(description="定向补充个人养老金参加流程内容（合规爬取）")
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
            print(f"  A: {r['answer'][:240]}...")
        return
    with open(OUT, "a", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已追加至 {OUT}")


if __name__ == "__main__":
    main()
