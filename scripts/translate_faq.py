#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/translate_faq.py — 英文金融 FAQ 数据集「选择性翻译」入库流水线

策略（对齐用户要求：每文件翻译一部分、覆盖全部领域、只译有价值的）：
  1. 对每个领域（= 每个源文件）按 type 分层抽样 --per-domain 条（默认 60），保证覆盖全部领域与题型。
  2. 调用 FinRag 自带 LLM 端点，对每批做「价值筛 + 翻译」二合一：
     - 筛：只保留对中文金融问答知识库真正有用/可检索的条目（金融知识、概念、产品、制度、政策、风控等）；
           剔除过于学术化、纯论文方法细节、仅讨论单篇论文、或与金融实务无关/空洞重复的条目。
     - 译：保留条目翻译为准确简体中文。
  3. 写出 ingest 兼容语料 data/crawled/faq_<domain>_qa.jsonl：
     category 钉到对应领域（Milvus 严格领域过滤可命中）、source_url 用伪协议保证 MySQL 幂等去重。
  4. --all 模式按 (source_pdf, chunk_id) 断点续传；抽样模式每次覆盖重写。

用法：
  python scripts/translate_faq.py                      # 每领域抽样 60 条 → 过滤+翻译
  python scripts/translate_faq.py --per-domain 100     # 每领域抽样 100 条
  python scripts/translate_faq.py --domain financial_markets,stock_market
  python scripts/translate_faq.py --all                # 全量（断点续传，较慢）
"""
import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import get_config

SRC_BASE = Path(r"D:\汪欢\data\faq")
CRAWLED_DIR = PROJECT_ROOT / "data" / "crawled"
CRAWLED_DIR.mkdir(parents=True, exist_ok=True)

# 领域 → (子目录, 文件名, 中文来源标签)
DOMAIN_FILES = {
    "banking":              ("Banking",                          "banking_qa_dataset.json", "银行FAQ"),
    "corporate_finance":    ("Corporate Finance",                "corporate_qa_dataset.json", "公司金融FAQ"),
    "financial_accounting": ("Financial Accounting",             "financialAccounting_qa_dataset.json", "财务会计FAQ"),
    "financial_markets":    ("Financial Markets & Institutions", "financialMarket_qa_dataset.json", "金融市场与机构FAQ"),
    "fintech":              ("FinTech & Digital Payments",       "FinTechDigitalPayments_qa_dataset.json", "金融科技与数字支付FAQ"),
    "insurance":            ("Insurance & Actuarial Finance",    "InsuranceActuarialFinance_qa_dataset.json", "保险与精算FAQ"),
    "investment_banking":    ("Investment_Banking",               "Investment_Banking_qa_dataset.json", "投资银行FAQ"),
    "personal_finance":      ("Personal Finance and Wealth Management", "PersonalFinanceWealthManagement_qa_dataset.json", "个人理财与财富管理FAQ"),
    "risk_management":       ("Risk Management",                 "RiskManagement_qa_dataset.json", "风险管理FAQ"),
    "stock_market":          ("Stock Market",                    "stock_qa_dataset.json", "股票市场FAQ"),
}

TYPE_MAP = {
    "factual": "事实性", "analytical": "分析型", "reasoning": "推理型",
    "scenario": "情景型", "hard": "综合型",
}
DEFAULT_TYPE = "综合型"

PROMPT = (
    "你是一名金融知识库审核与翻译专家。下面是一批英文金融问答，每条带序号 idx、question、answer。\n"
    "请完成两步：\n"
    "1) 筛选：只保留对「中文金融问答知识库」真正有价值、实用、可检索的条目——即面向投资者/从业者/公众的金融知识、"
    "概念、产品、制度、政策、风险管理、市场运作等。剔除：过于学术化或纯论文方法细节、仅讨论单篇论文结论、"
    "与金融实务无关、或明显重复空洞的条目。\n"
    "2) 翻译：将保留的条目准确翻译为简体中文，保留专业术语（如 IPO、VaR、Basel III 可括号注中文）。\n"
    "3) 输出：严格只返回一个 JSON 数组，每个元素为 {\"idx\": 原序号, \"question\": 中文问题, \"answer\": 中文答案}。"
    "只返回数组，不要任何额外说明或 markdown 代码块。若本批无有用条目，返回空数组 []。"
)


def _translate_batch(records: list[dict], cfg, batch_size: int, max_workers: int) -> list[dict]:
    """records: [{idx, source_pdf, chunk_id, question, answer, type}]
    返回筛选+翻译后的中文记录（含溯源字段）。"""
    api_key = cfg.llm.api_key
    base = cfg.llm.base_url.rstrip("/")
    model = cfg.llm.model
    url = base + "/chat/completions"

    payload = [{"idx": r["idx"], "question": r["question"], "answer": r["answer"]} for r in records]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT + "\n" + json.dumps(payload, ensure_ascii=False)}],
        "temperature": 0.1,
        "max_tokens": max(1200, batch_size * 500),
        "enable_thinking": False,
    }).encode("utf-8")

    last_err = None
    for attempt in range(4):
        try:
            import urllib.request
            req = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read().decode("utf-8"))
            content = resp["choices"][0]["message"]["content"]
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            translated = json.loads(content)
            if not isinstance(translated, list):
                translated = []
            # 按 idx 映射回源记录
            by_idx = {r["idx"]: r for r in records}
            out = []
            for t in translated:
                i = t.get("idx")
                src = by_idx.get(i)
                if src is None:
                    continue
                q = (t.get("question") or "").strip()
                a = (t.get("answer") or "").strip()
                if not q or not a or len(a) < 20:
                    continue
                out.append({
                    "source_pdf": src["source_pdf"], "chunk_id": src["chunk_id"],
                    "type": src["type"], "question": q, "answer": a,
                })
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            logger.warning(f"翻译批次失败(尝试 {attempt+1}/4): {e}，{wait}s 后重试")
            time.sleep(wait)
    logger.error(f"翻译批次最终失败: {last_err}，本批跳过（可重跑补全）")
    return []


def _stratified_sample(records: list[dict], per_domain: int) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r.get("type", DEFAULT_TYPE), []).append(r)
    types = list(by_type.keys())
    if per_domain >= len(records) or not types:
        return records
    per_type = max(1, per_domain // len(types))
    sampled: list[dict] = []
    for t in types:
        items = by_type[t]
        random.shuffle(items)
        sampled.extend(items[:per_type])
    if len(sampled) < per_domain:
        remaining = [r for r in records if r not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[: per_domain - len(sampled)])
    random.shuffle(sampled)
    return sampled[:per_domain]


def _load_raw(domain: str, src: Path) -> list[dict]:
    _, fname, _ = DOMAIN_FILES[domain]
    path = src / DOMAIN_FILES[domain][0] / fname
    if not path.exists():
        logger.error(f"原始文件不存在，跳过 {domain}: {path}")
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and isinstance(data.get("train"), list):
        data = data["train"]
    out = []
    for i, item in enumerate(data):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a or len(a) < 20:
            continue
        out.append({
            "idx": i, "source_pdf": item.get("source_pdf", ""), "chunk_id": item.get("chunk_id", ""),
            "question": q, "answer": a, "type": item.get("type", ""),
        })
    return out


def translate_domain(domain: str, src: Path, per_domain: int, do_all: bool,
                     batch_size: int, max_workers: int, out_suffix: str = "") -> int:
    _, _, label = DOMAIN_FILES[domain]
    raw = _load_raw(domain, src)
    if not raw:
        return 0
    suffix = f"_{out_suffix}" if out_suffix else ""
    out_path = CRAWLED_DIR / f"faq_{domain}{suffix}_qa.jsonl"

    if do_all:
        done = set()
        if out_path.exists():
            for line in open(out_path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("_key"):
                    done.add(o["_key"])
        logger.info(f"[{domain}] 断点续传：已译 {len(done)} 条，源 {len(raw)} 条")
        to_translate = [r for r in raw if f"{r['source_pdf']}#{r['chunk_id']}" not in done]
        mode = "a"
    else:
        to_translate = _stratified_sample(raw, per_domain)
        mode = "w"

    if not to_translate:
        logger.info(f"[{domain}] 无需翻译")
        return 0

    batches = [to_translate[i:i + batch_size] for i in range(0, len(to_translate), batch_size)]
    cfg = get_config()
    results: list = [None] * len(batches)
    done_batches = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_translate_batch, b, cfg, batch_size, max_workers): i for i, b in enumerate(batches)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done_batches += 1
            if done_batches % 5 == 0 or done_batches == len(batches):
                logger.info(f"[{domain}] 翻译进度 {done_batches}/{len(batches)} 批")

    count = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for batch in results:
            if not batch:
                continue
            for r in batch:
                rec = {
                    "category": domain,
                    "question": r["question"],
                    "answer": r["answer"],
                    "type": TYPE_MAP.get(r["type"].lower(), DEFAULT_TYPE) if r["type"] else DEFAULT_TYPE,
                    "source_name": label,
                    "source_url": f"faq://{domain}/{r['source_pdf']}#{r['chunk_id']}",
                    "_key": f"{r['source_pdf']}#{r['chunk_id']}",
                    "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    logger.info(f"[{domain}] 完成，本批写出 {count} 条（保留率 {count}/{len(to_translate)}）→ {out_path.name}")
    return count


def main():
    ap = argparse.ArgumentParser(description="英文 FAQ 选择性翻译（筛+译）")
    ap.add_argument("--src", type=str, default=str(SRC_BASE))
    ap.add_argument("--per-domain", type=int, default=60, help="每领域抽样条数（默认 60）")
    ap.add_argument("--all", action="store_true", help="全量翻译（断点续传，较慢）")
    ap.add_argument("--domain", type=str, default="", help="只处理指定领域，逗号分隔")
    ap.add_argument("--batch", type=int, default=12, help="每批条数")
    ap.add_argument("--workers", type=int, default=6, help="并发批数")
    ap.add_argument("--out-suffix", type=str, default="", help="输出文件后缀，如 _extra → faq_<域>_extra_qa.jsonl（不覆盖主文件）")
    args = ap.parse_args()

    src = Path(args.src)
    domains = args.domain.split(",") if args.domain else list(DOMAIN_FILES.keys())
    domains = [d.strip() for d in domains if d.strip() in DOMAIN_FILES]
    if not domains:
        logger.error("没有可处理的领域")
        return

    logger.info("=" * 60)
    logger.info("英文 FAQ 选择性翻译（覆盖全部领域 + 价值筛选）")
    logger.info(f"  模式: {'全量' if args.all else f'每领域抽样 {args.per_domain} 条'}")
    logger.info(f"  领域: {domains}")
    logger.info("=" * 60)

    total = 0
    for d in domains:
        total += translate_domain(d, src, args.per_domain, args.all, args.batch, args.workers, args.out_suffix)
    logger.info(f"全部完成，累计写出 {total} 条（已过滤无价值条目）")


if __name__ == "__main__":
    main()
