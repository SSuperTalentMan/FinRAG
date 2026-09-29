#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/bm25.py — BM25 高频问答匹配服务
基于 jieba 分词和 rank_bm25 实现 MySQL FAQ 的精确匹配。
"""

import time
import threading

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from loguru import logger
from db.mysql import get_all_qa_for_bm25

# ─── 中文分词器（带金融领域增强）─────────────────────────────────────────────────
# 金融领域自定义词典，提升分词质量
_FINANCIAL_TERMS = [
    "银行", "信贷", "贷款", "利率", "汇率", "证券", "基金", "股票", "债券",
    "财务", "会计", "审计", "资产负债", "现金流", "折旧", "资产负债表",
    "公司金融", "企业金融", "融资", "股权", "并购", "并购重组",
    "风险管理", "保险", "精算", "再保险",
    "外汇", "同业拆借", "流动性", "资本充足率",
    "IPO", "科创板", "纳斯达克", "道琼斯", "标普500",
    "金融科技", "区块链", "数字货币", "央行数字货币",
    "财富管理", "私人银行", "资产配置", "投资组合",
    "尽职调查", "估值", "市盈率", "市净率",
]

for term in _FINANCIAL_TERMS:
    jieba.add_word(term)


def tokenize(text: str) -> list[str]:
    """中文分词，保留金融术语不拆分。"""
    # 先提取英文/数字序列（整体保留）
    tokens = []
    for word in jieba.cut(text):
        word = word.strip()
        if word:
            tokens.append(word)
    return tokens


class BM25Retriever:
    """
    BM25 检索器。
    在 MySQL finance_faq 中构建 BM25 索引，支持按领域过滤。

    并发一致性：索引状态保存在单一元组 _index_state 中，检索线程一次读取整个快照，
    重建线程构建完成后一次性替换，避免出现"新文档列表 + 旧索引"的错位组合
    （两者条数不一致时 get_scores 会越界或错误打分）。
    _docs / _bm25 等旧属性名以 property 形式保留，兼容测试与既有调用方直接读写。
    """

    def __init__(self):
        # (bm25, docs, bm25_by_domain, docs_by_domain) 不可变快照，整体替换
        self._index_state: tuple | None = None
        self._loaded = False
        self._last_built = 0.0
        self._lock = threading.Lock()  # 重建索引时互斥，避免并发重复构建
        # 索引自动刷新间隔（秒）：FAQ 数据变更后，最多延迟该时长感知到新数据
        self._refresh_interval = 300

    # ── 兼容属性：内部统一走 _index_state，保证读写的一致性 ──
    @property
    def _bm25(self) -> BM25Okapi | None:
        return self._index_state[0] if self._index_state else None

    @_bm25.setter
    def _bm25(self, value: BM25Okapi | None) -> None:
        cur = self._index_state or (None, [], {}, {})
        self._index_state = (value, cur[1], cur[2], cur[3])

    @property
    def _docs(self) -> list[dict]:
        return self._index_state[1] if self._index_state else []

    @_docs.setter
    def _docs(self, value: list[dict]) -> None:
        cur = self._index_state or (None, [], {}, {})
        self._index_state = (cur[0], value, cur[2], cur[3])

    @property
    def _bm25_by_domain(self) -> dict[str, BM25Okapi]:
        return self._index_state[2] if self._index_state else {}

    @_bm25_by_domain.setter
    def _bm25_by_domain(self, value: dict[str, BM25Okapi]) -> None:
        cur = self._index_state or (None, [], {}, {})
        self._index_state = (cur[0], cur[1], value, cur[3])

    @property
    def _docs_by_domain(self) -> dict[str, list[dict]]:
        return self._index_state[3] if self._index_state else {}

    @_docs_by_domain.setter
    def _docs_by_domain(self, value: dict[str, list[dict]]) -> None:
        cur = self._index_state or (None, [], {}, {})
        self._index_state = (cur[0], cur[1], cur[2], value)

    def _load(self) -> None:
        """懒加载 + 定时刷新：从 MySQL 读取所有 FAQ，构建 BM25 索引。"""
        if self._loaded and (time.time() - self._last_built) < self._refresh_interval:
            return
        with self._lock:
            # 双检：锁内再次判断，避免多个线程同时重建
            if self._loaded and (time.time() - self._last_built) < self._refresh_interval:
                return
            self._rebuild()

    def _rebuild(self) -> None:
        """（重）构建 BM25 索引（全量 + 按 domain 分桶）。失败时保留旧索引。"""
        try:
            items = get_all_qa_for_bm25()
            if not items:
                logger.warning("MySQL 中无 FAQ 数据，BM25 无法构建")
                return
            # 全部在局部变量中构建完成后，一次性写入快照（原子替换，见类注释）
            tokenized = [tokenize(d["question"]) for d in items]
            new_bm25 = BM25Okapi(tokenized)

            # 按 domain 预分桶：检索时直接定位对应桶，避免全表遍历 + 逐条过滤
            buckets: dict[str, list[dict]] = {}
            bucket_tokens: dict[str, list[list[str]]] = {}
            for d in items:
                cat = d.get("category", "general")
                buckets.setdefault(cat, []).append(d)
                bucket_tokens.setdefault(cat, []).append(tokenize(d["question"]))
            new_bm25_by_domain = {
                cat: BM25Okapi(toks) for cat, toks in bucket_tokens.items() if toks
            }

            self._index_state = (new_bm25, items, new_bm25_by_domain, buckets)
            self._loaded = True
            self._last_built = time.time()
            logger.info(
                f"BM25 索引构建完成，共 {len(items)} 条问题，"
                f"{len(new_bm25_by_domain)} 个领域桶: {list(new_bm25_by_domain.keys())}"
            )
        except Exception as e:
            logger.error(f"BM25 索引构建失败（保留旧索引）: {e}")

    def search(self, query: str, domain: str | None = None, top_k: int = 10) -> list[dict]:
        """
        在 BM25 索引中搜索，可选按领域过滤。
        指定 domain 时直接定位对应桶，避免全表打分 + 逐条过滤。
        返回 [ {"question", "answer", "category", "score", "source"} ]
        """
        self._load()
        # 一次读取完整快照，保证 bm25 与 docs 严格配对（与重建线程并发时不错位）
        state = self._index_state
        if state is None or state[0] is None or not state[1]:
            return []
        bm25, docs, bm25_by_domain, docs_by_domain = state

        tokens = tokenize(query)

        # 域分桶检索：有 domain 时用桶索引，避免全量打分
        if domain and domain in bm25_by_domain:
            bm25 = bm25_by_domain[domain]
            docs = docs_by_domain[domain]

        scores = bm25.get_scores(tokens)
        results = []
        for i, score in enumerate(scores):
            doc = docs[i]
            # 桶索引已按 domain 过滤；全量索引下需要逐条过滤
            if domain and doc["category"] != domain:
                continue
            results.append({
                "question": doc["question"],
                "answer":   doc["answer"],
                "category": doc["category"],
                "type":     doc.get("type", ""),
                "score":    float(score),
                "source":   doc.get("source", ""),
            })

        # softmax 归一化分数，便于阈值判断
        if results:
            scores_arr = np.array([r["score"] for r in results])
            exp_scores = np.exp(scores_arr - scores_arr.max())
            probs = (exp_scores / exp_scores.sum()).tolist()
            for r, p in zip(results, probs):
                r["softmax_score"] = round(p, 6)

        results.sort(key=lambda x: x["softmax_score"], reverse=True)
        return results[:top_k]

    def get_best_match(self, query: str, domain: str | None = None,
                       threshold: float = 0.7) -> dict | None:
        """
        获取 BM25 最高匹配结果，若 softmax 分数低于 threshold 则返回 None。
        """
        results = self.search(query, domain=domain, top_k=1)
        if not results:
            return None
        best = results[0]
        if best["softmax_score"] >= threshold:
            logger.info(f"BM25 命中！domain={best['category']} score={best['softmax_score']:.4f}")
            return best
        logger.debug(f"BM25 未命中（低于阈值 {threshold}），score={best['softmax_score']:.4f}")
        return None


# 全局单例
_bm25_retriever: BM25Retriever | None = None
_bm25_lock = threading.Lock()  # 双检锁：避免并发首请求重复创建实例


def get_bm25_retriever() -> BM25Retriever:
    """获取 BM25 检索器单例（线程安全）。"""
    global _bm25_retriever
    if _bm25_retriever is None:
        with _bm25_lock:
            if _bm25_retriever is None:
                _bm25_retriever = BM25Retriever()
    return _bm25_retriever
