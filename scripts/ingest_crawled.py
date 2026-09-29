#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/ingest_crawled.py — 将合规爬取的公开金融问答数据入库

处理流程：
  1. 读取 data/crawled/*.jsonl
  2. 用意图分类器的关键词模块把记录映射到 11 个既有领域（category），
     保证 Milvus 领域过滤检索（category == domain）能命中；无命中归入 general。
  3. 写入 MySQL finance_faq（增量，不删除任何旧数据）
  4. BGE-M3 编码后增量写入 Milvus finrag_faq（不清旧数据）

用法：
    python scripts/ingest_crawled.py
    python scripts/ingest_crawled.py --mysql-only
"""
import os
import sys
import json
import glob
import argparse
from pathlib import Path
from loguru import logger

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
_cfg = get_config()

import pymysql
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel
from scipy import sparse as sp_sparse

from rag_qa.core.query_classifier import _keyword_classify

CRAWLED_DIR = PROJECT_ROOT / "data" / "crawled"
SPARSE_DIM = 250002

# 来源类型 → 中文题型
TYPE_BY_SOURCE = {
    "csrc": "事实性",   # 监管规则问答
    "sipf": "情景型",   # 投资者保护情景
    "sse":  "综合型",   # 投教知识文章
    "gov":  "分析型",   # 宏观政策解读
}

# ─── 宏观政策（gov.cn）专用领域映射 ─────────────────────────────────────────────
# 政策文件的措辞与日常提问差异大，直接用通用关键词分类大量落入 general；
# 而 Milvus 检索对 domain 是严格过滤（category == domain），general 只在
# 意图为 general 时才被检索，因此必须为政策文本单独建一套映射词表。
MACRO_KEYWORDS: dict[str, list[str]] = {
    "banking": [
        "银行", "信贷", "存款", "贷款", "准备金", "利率市场化", "LPR", "普惠金融",
        "小微", "涉农贷款", "不良贷款", "资本充足率", "政策性银行", "商业银行",
        "支付清算", "网点", "柜台", "开户", "账户", "征信", "担保", "融资担保",
    ],
    "insurance": [
        "保险", "承保", "理赔", "保费", "偿付能力", "再保险", "农业保险",
        "大病保险", "养老", "年金", "医保", "社保", "长期护理", "保险资金",
    ],
    "stock_market": [
        "股票", "股市", "上市公司", "IPO", "注册制", "退市", "投资者",
        "证券交易所", "科创板", "创业板", "北交所", "信息披露", "分红", "市值管理",
    ],
    "financial_markets": [
        "货币政策", "利率", "汇率", "债券", "国债", "地方政府债", "专项债",
        "货币市场", "外汇", "人民币国际化", "金融开放", "金融体系", "资本市场",
        "直接融资", "间接融资", "人民币", "流动性", "金融基础设施", "金融供给侧",
    ],
    "corporate_finance": [
        "企业融资", "融资成本", "民营企业", "中小企业", "减税降费", "营商环境",
        "股权投资", "创业投资", "企业上市", "产权", "混合所有制", "国企改革",
    ],
    "fintech": [
        "数字人民币", "金融科技", "移动支付", "数字货币", "跨境支付", "数据要素",
        "数字化", "区块链", "人工智能", "平台经济",
    ],
    "risk_management": [
        "风险", "防范化解", "地方债务", "影子银行", "非法集资", "处置", "监管",
        "合规", "压力测试", "系统性", "底线",
    ],
    "personal_finance": [
        "居民收入", "居民消费", "储蓄", "理财", "个人养老金", "住房公积",
        "消费信贷", "工资", "就业", "民生",
    ],
    "investment_banking": [
        "承销", "并购重组", "资产证券化", "REITs", "基础设施投资", "政府引导基金",
    ],
    "financial_accounting": [
        "会计", "财务", "审计", "会计准则", "财务报告", "预算", "决算", "财政资金",
    ],
}

# 命中后仍判不出领域时，若文本属于泛财经范畴则归入该兜底领域
MACRO_FALLBACK = "financial_markets"
MACRO_FIN_RELEVANT = ("金融", "经济", "财政", "税", "投资", "资本", "市场", "货币")
VALID_DOMAINS = set(MACRO_KEYWORDS) | {"general"}


def map_macro_domain(question: str, answer: str) -> str:
    """
    将 gov.cn 宏观政策文本映射到 11 个既有领域之一。
    计分：标题/问题权重 3，答案前 1500 字权重 1（按命中次数累加）。
    """
    head = (question or "")[:200]
    body = (answer or "")[:1500]
    scores: dict[str, int] = {}
    for domain, kws in MACRO_KEYWORDS.items():
        s = 0
        for kw in kws:
            if kw in head:
                s += 3
            n = body.count(kw)
            if n:
                s += min(n, 3)
        if s:
            scores[domain] = s
    if not scores:
        text = head + body
        if any(k in text for k in MACRO_FIN_RELEVANT):
            return MACRO_FALLBACK
        return "general"
    best = max(scores, key=scores.get)
    # 次优领域得分接近时不强行判定，交给兜底
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) > 1 and ranked[0] - ranked[1] <= 1 and scores[best] < 6:
        return MACRO_FALLBACK if scores[best] >= 3 else "general"
    return best


def load_crawled(only: str | None = None) -> list[dict]:
    rows = []
    seen_q: set[str] = set()
    for p in sorted(glob.glob(str(CRAWLED_DIR / "*_qa.jsonl"))):
        src = Path(p).stem.replace("_qa", "")
        if only and src != only:
            continue
        ttype = TYPE_BY_SOURCE.get(src, "综合型")
        n = 0
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (o.get("question") or "").strip()
            a = (o.get("answer") or "").strip()
            if not q or not a or len(a) < 30:
                continue
            if q in seen_q:          # 跨来源 / 同来源的重复问题只留一条
                continue
            seen_q.add(q)
            # 领域映射：gov.cn 宏观政策走专用映射，其余走通用关键词分类
            # 若爬取记录自带合法 category（如定向补充脚本显式钉到 stock_market），
            # 优先采用，避免 _keyword_classify 把「分红/上市公司」误分 general 导致检索不可见。
            provided = o.get("category")
            if provided in VALID_DOMAINS:
                dom = provided
            elif src == "gov":
                dom = map_macro_domain(q, a)
            else:
                dom, _, _ = _keyword_classify(q)
                dom = dom if dom in VALID_DOMAINS else "general"
            rows.append({
                "category": dom if dom in VALID_DOMAINS else "general",
                "question": q,
                "answer": a,
                "type": ttype,
                "source": o.get("source_name") or (o.get("source_url") or src),
                "source_url": o.get("source_url", ""),
                "crawled_at": o.get("crawled_at", ""),
            })
            n += 1
        logger.info(f"  {src}: {n} 条")
    return rows


def insert_mysql(rows: list[dict]) -> list[dict]:
    """写入 MySQL 并返回真正新增的行（供 Milvus 复用，避免向量重复）。"""
    logger.info("写入 MySQL finance_faq（增量）...")
    conn = pymysql.connect(
        host=_cfg.mysql.host, port=_cfg.mysql.port, user=_cfg.mysql.user,
        password=_cfg.mysql.password, database=_cfg.mysql.database, charset=_cfg.mysql.charset,
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS finance_faq (
        id INT AUTO_INCREMENT PRIMARY KEY, category VARCHAR(50) NOT NULL,
        question VARCHAR(1000) NOT NULL, answer TEXT NOT NULL,
        source VARCHAR(500) DEFAULT '', type VARCHAR(50) DEFAULT '',
        source_url VARCHAR(500) DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        KEY idx_category (category), KEY idx_question (question(191))) 
        ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""")
    # 兼容旧表：补充 source_url 列
    cur.execute("SHOW COLUMNS FROM finance_faq LIKE 'source_url'")
    if not cur.fetchone():
        cur.execute("ALTER TABLE finance_faq ADD COLUMN source_url VARCHAR(500) DEFAULT ''")
        conn.commit()
        logger.info("已为 finance_faq 补充 source_url 列")
    # 幂等：按 (source_url, question) 去重（同一页面多组问答共享 URL，不能只按 URL 去重）
    cur.execute("SELECT source_url, question FROM finance_faq WHERE source_url != ''")
    existing = {(r["source_url"], r["question"]) for r in cur.fetchall()}
    inserted_rows: list[dict] = []
    for r in rows:
        if r["source_url"] and (r["source_url"], r["question"]) in existing:
            continue
        cur.execute(
            """INSERT INTO finance_faq (category, question, answer, source, type, source_url)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (r["category"], r["question"], r["answer"], r["source"][:200],
             r["type"], r["source_url"][:500]),
        )
        existing.add((r["source_url"], r["question"]))
        inserted_rows.append(r)
    conn.commit()
    cur.execute("SELECT category, COUNT(*) c FROM finance_faq GROUP BY category ORDER BY category")
    dist = cur.fetchall()
    conn.close()
    logger.info(f"MySQL 新增 {len(inserted_rows)} 条（跳过重复 {len(rows) - len(inserted_rows)} 条），当前分布：")
    for row in dist:
        logger.info(f"  {row['category']:22s} : {row['c']}")
    return inserted_rows


def insert_milvus(rows: list[dict], batch_size: int = 96):
    logger.info("写入 Milvus finrag_faq（增量，BGE-M3 编码）...")
    kw = {"uri": _cfg.milvus.uri, "db_name": _cfg.milvus.database_name}
    if _cfg.milvus.token:
        kw["token"] = _cfg.milvus.token
    client = MilvusClient(**kw)
    if not client.has_collection(_cfg.milvus.collection_name):
        logger.error("集合不存在，请先运行 load_new_data.py 初始化")
        return
    model = BGEM3FlagModel(str(Path(_cfg.bge_m3_path)), use_fp16=False)

    texts = [f"{r['question']} {r['answer']}" for r in rows]
    dense_all, sparse_all = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = model.encode(batch, batch_size=batch_size, return_dense=True,
                           return_sparse=True, return_colbert_vecs=False)
        dense_all.extend(res["dense_vecs"].tolist())
        for lexical in res["lexical_weights"]:
            if lexical:
                idxs = [int(k) for k in lexical.keys()]
                vals = [float(v) for v in lexical.values()]
                sparse_all.append(sp_sparse.csr_matrix(
                    (vals, ([0] * len(idxs), idxs)), shape=(1, max(idxs) + 1)))
            else:
                sparse_all.append(sp_sparse.csr_matrix((1, SPARSE_DIM)))

    # 幂等：删除本次 source_url 已有向量（按 source 字段精确匹配较难，改为整批按 question 去重前先清理重复插入风险 → 直接插入，靠 source_url 标记人工核对）
    data = [{
        "dense_vector": dense_all[j], "sparse_vector": sparse_all[j],
        "text": f"{r['question']}\n答：{r['answer']}",
        "category": r["category"], "type": r["type"],
        "question": r["question"], "answer": r["answer"],
    } for j, r in enumerate(rows)]
    total = 0
    for i in range(0, len(data), batch_size):
        client.insert(collection_name=_cfg.milvus.collection_name, data=data[i:i + batch_size])
        total += len(data[i:i + batch_size])
    client.load_collection(_cfg.milvus.collection_name)
    stats = client.get_collection_stats(_cfg.milvus.collection_name)
    logger.info(f"Milvus 新增 {total} 条，集合总行数 {stats.get('row_count')}")


def main():
    ap = argparse.ArgumentParser(description="爬取数据入库")
    ap.add_argument("--mysql-only", action="store_true")
    ap.add_argument("--only", default=None,
                    help="只处理指定来源，如 --only gov（csrc/sipf/sse/gov）")
    ap.add_argument("--dry-run", action="store_true", help="只统计领域分布，不写库")
    args = ap.parse_args()
    logger.info("=" * 60)
    logger.info("爬取数据入库（映射到既有领域）")
    logger.info("=" * 60)
    rows = load_crawled(only=args.only)
    logger.info(f"共加载 {len(rows)} 条爬取数据")
    if not rows:
        logger.warning("没有可入库的数据")
        return
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["category"]] = dist.get(r["category"], 0) + 1
    logger.info("领域分布预览：")
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        logger.info(f"  {k:22s} : {v}")
    if args.dry_run:
        logger.info("--dry-run：跳过写库")
        return
    new_rows = insert_mysql(rows)
    if not args.mysql_only:
        if not new_rows:
            logger.info("MySQL 无新增，跳过 Milvus 写入（避免向量重复）")
            return
        insert_milvus(new_rows)


if __name__ == "__main__":
    main()
