#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/reconciliation.py — MySQL↔Milvus 定时数据对账

背景：Milvus 与 MySQL 是两套独立存储，删除/重建过程中可能残留孤儿向量
（教训：Milvus 在数据重载/清理后可能残留孤儿向量，需定期对账）。

同一 collection 混存两类数据：
  1. FAQ 问答行（无 kb_id，question 字段 = 问题）
  2. 文档块行（带 kb_id / doc_id）

对账逻辑：
  - FAQ「缺失」：MySQL finance_faq 的问题在 Milvus 无向量（真实问题但检索不到）→ 告警
  - FAQ「多余」：Milvus 多出的 FAQ 属设计内（translated/crawled 全量索引），仅统计不清理
  - 孤儿文档向量：Milvus 行 doc_id 在 MySQL documents 中已不存在（删除遗留）→ 可清理
  - 孤儿知识库向量：Milvus 行 kb_id 在 knowledge_bases 中已不存在（级联删除遗漏）→ 可清理
  - 缺失文档向量：MySQL status=indexed 但 Milvus 无对应行 → 告警

行为：
  - 定时执行（config.reconciliation.sync_interval，0=关闭），后台 daemon 线程
  - 结果写入审计（recon.run）+ 结构化日志；差异数超过 alert_threshold 升级告警
  - auto_clean=True 时按主键删除「确定性孤儿」（文档/知识库向量）；FAQ 多余向量不清理
任何异常 fail-open：只记日志，不抛出、不影响业务。
"""

import threading
import time
from typing import Optional

from loguru import logger

from config import get_config
from db.mysql import get_db
from db.milvus import get_milvus_client
from services.audit import record_audit

# 单次对账拉取 Milvus 行的分页大小（避免一次性载入过多内存）
_PAGE_SIZE = 2000
# 删除时按主键分批的大小
_DELETE_BATCH = 500


# ─── MySQL 侧快照 ──────────────────────────────────────────────────────────────
def _mysql_faq_questions() -> set[str]:
    """MySQL 中全部 FAQ 问题集合。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT question FROM finance_faq")
        return {str(r["question"]).strip() for r in cur.fetchall() if r.get("question")}


def _mysql_indexed_docs() -> dict[int, int]:
    """MySQL 中 status=indexed 的文档 {doc_id: kb_id}。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, kb_id FROM documents WHERE status='indexed'")
        return {int(r["id"]): int(r["kb_id"]) for r in cur.fetchall()}


def _mysql_kb_ids() -> set[int]:
    """MySQL 中全部知识库 id 集合。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM knowledge_bases")
        return {int(r["id"]) for r in cur.fetchall()}


# ─── Milvus 侧扫描 ─────────────────────────────────────────────────────────────
def _iter_milvus_rows(output_fields: list[str]):
    """分页遍历 Milvus collection 全部行（避免一次载入过多内存）。"""
    cfg = get_config()
    client = get_milvus_client()
    offset = 0
    while True:
        rows = client.query(
            collection_name=cfg.milvus.collection_name,
            filter="",
            limit=_PAGE_SIZE,
            offset=offset,
            output_fields=output_fields,
        )
        if not rows:
            break
        yield from rows
        offset += len(rows)
        if len(rows) < _PAGE_SIZE:
            break


def _delete_by_ids(ids: list[int]) -> int:
    """按主键批量删除 Milvus 行，返回删除条数。"""
    if not ids:
        return 0
    cfg = get_config()
    client = get_milvus_client()
    deleted = 0
    for i in range(0, len(ids), _DELETE_BATCH):
        batch = ids[i:i + _DELETE_BATCH]
        client.delete(collection_name=cfg.milvus.collection_name, ids=batch)
        deleted += len(batch)
    return deleted


# ─── 对账主体 ──────────────────────────────────────────────────────────────────
def run_reconciliation(auto_clean: Optional[bool] = None) -> dict:
    """执行一次 MySQL↔Milvus 对账，返回结果摘要；不会抛出异常（fail-open）。

    外部依赖（MySQL/Milvus）异常时记录告警并返回含 error 字段的空结果，
    保证定时任务/调用方不受影响。
    """
    try:
        return _do_reconcile(auto_clean)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"对账执行失败（fail-open，不影响业务）: {e}")
        return {"error": str(e), "auto_clean": bool(auto_clean)}


def _do_reconcile(auto_clean: Optional[bool]) -> dict:
    """对账核心逻辑（被 run_reconciliation 安全包装）。"""
    cfg = get_config()
    if auto_clean is None:
        auto_clean = cfg.reconciliation.auto_clean
    alert_threshold = cfg.reconciliation.alert_threshold
    start = time.time()

    # 1) MySQL 侧快照
    mysql_faq_qs = _mysql_faq_questions()
    mysql_docs = _mysql_indexed_docs()
    kb_ids = _mysql_kb_ids()

    # 2) 扫描 Milvus 全量行，分类 FAQ / 文档块
    faq_question_ids: dict[str, list[int]] = {}   # question -> [pk ids]
    milvus_faq_total = 0
    milvus_doc_chunk_total = 0
    no_doc_id_total = 0                            # 旧数据（有 kb 无 doc_id），仅统计不处理
    doc_entries: dict[tuple[int, int], list[int]] = {}  # (kb_id, doc_id) -> [pk ids]
    for row in _iter_milvus_rows(["id", "kb_id", "doc_id", "question"]):
        pk = int(row["id"])
        if row.get("kb_id") is None:
            # FAQ 行：无 kb_id
            q = str(row.get("question") or "").strip()
            if q:
                faq_question_ids.setdefault(q, []).append(pk)
                milvus_faq_total += 1
            continue
        # 文档块行：带 kb_id
        milvus_doc_chunk_total += 1
        kb_id = int(row["kb_id"])
        doc_id = row.get("doc_id")
        if doc_id is None:
            no_doc_id_total += 1
            continue
        doc_entries.setdefault((kb_id, int(doc_id)), []).append(pk)

    # 3) 差异判定
    # FAQ 缺失：MySQL 有、Milvus 无（检索不到的真问题）
    missing_faq = sorted(mysql_faq_qs - set(faq_question_ids))

    # 孤儿文档向量 / 孤儿知识库向量：可确定性判定，允许清理
    orphan_doc_ids: list[int] = []      # 已不存在的 doc_id
    orphan_doc_vectors = 0
    orphan_kb_ids: list[int] = []       # 已不存在的 kb_id
    orphan_kb_vectors = 0
    mysql_doc_set = set(mysql_docs)
    for (kb_id, doc_id), ids in doc_entries.items():
        if kb_id not in kb_ids:
            orphan_kb_ids.append(kb_id)
            orphan_kb_vectors += len(ids)
        elif doc_id not in mysql_doc_set:
            orphan_doc_ids.append(doc_id)
            orphan_doc_vectors += len(ids)

    # 缺失文档向量：MySQL 已索引但 Milvus 无对应行
    milvus_doc_set = set(doc_id for _, doc_id in doc_entries)
    missing_doc_ids = sorted(mysql_doc_set - milvus_doc_set)

    # 4) 自动清理（仅确定性孤儿；FAQ 多余属设计内，永不清理）
    cleaned_doc = cleaned_kb = 0
    if auto_clean and (orphan_doc_ids or orphan_kb_ids):
        try:
            # 孤儿知识库向量：kb_id 已不存在（级联删除遗漏）
            cleaned_kb = _delete_by_ids(
                [pk for (kb_id, _), ids in doc_entries.items() if kb_id not in kb_ids for pk in ids]
            )
            # 孤儿文档向量：文档已删除但向量残留
            cleaned_doc = _delete_by_ids(
                [pk for (kb_id, doc_id), ids in doc_entries.items()
                 if kb_id in kb_ids and doc_id not in mysql_doc_set for pk in ids]
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"对账自动清理失败（不影响结果上报）: {e}")

    # 5) 汇总 + 日志 + 审计
    elapsed = round(time.time() - start, 3)
    total_diff = len(missing_faq) + len(orphan_doc_ids) + len(orphan_kb_ids) + len(missing_doc_ids)
    result = {
        "faq_mysql":          len(mysql_faq_qs),
        "faq_milvus":         milvus_faq_total,
        "faq_missing":        len(missing_faq),
        "doc_mysql_indexed":  len(mysql_docs),
        "doc_milvus_chunks":  milvus_doc_chunk_total,
        "doc_no_doc_id":      no_doc_id_total,
        "orphan_docs":        orphan_doc_ids[:50],       # 截断，避免日志/审计过大
        "orphan_doc_vectors": orphan_doc_vectors,
        "orphan_kbs":         orphan_kb_ids[:50],
        "orphan_kb_vectors":  orphan_kb_vectors,
        "missing_docs":       missing_doc_ids[:50],
        "cleaned_doc_vectors": cleaned_doc,
        "cleaned_kb_vectors":  cleaned_kb,
        "elapsed_seconds":    elapsed,
        "auto_clean":         auto_clean,
    }

    msg = (
        f"对账完成: FAQ(MySQL {result['faq_mysql']}/Milvus {result['faq_milvus']}, 缺失 {result['faq_missing']}) | "
        f"文档块(MySQL {result['doc_mysql_indexed']}/Milvus {result['doc_milvus_chunks']}) | "
        f"孤儿文档 {len(result['orphan_docs'])} ({result['orphan_doc_vectors']} 向量) | "
        f"孤儿KB {len(result['orphan_kbs'])} ({result['orphan_kb_vectors']} 向量) | "
        f"缺失文档 {len(result['missing_docs'])} | 清理 doc={cleaned_doc}, kb={cleaned_kb} | {elapsed}s"
    )
    if total_diff > 0:
        detail = msg
        if total_diff >= alert_threshold:
            logger.warning(f"对账告警(差异 {total_diff} ≥ 阈值 {alert_threshold}): {msg}")
        else:
            logger.info(f"对账发现差异(共 {total_diff}): {msg}")
    else:
        logger.info(msg)
    try:
        record_audit(
            None, "system", "recon.run",
            target_type="system", target_id="",
            detail=msg, success=(total_diff == 0),
        )
    except Exception:  # noqa: BLE001
        pass
    return result


# ─── 定时任务 ──────────────────────────────────────────────────────────────────
def start_reconciliation_loop(interval: int) -> None:
    """启动后台定时对账线程（daemon，随进程退出）。interval 秒间隔，0=不启动。"""
    if interval <= 0:
        return
    cfg = get_config()

    def _loop():
        logger.info(f"数据对账定时任务已启动: 间隔 {interval}s, auto_clean={cfg.reconciliation.auto_clean}")
        while True:
            time.sleep(interval)
            try:
                run_reconciliation()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"定时对账执行失败（下轮重试）: {e}")

    threading.Thread(target=_loop, daemon=True, name="reconciliation-loop").start()
