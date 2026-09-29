# -*- coding: utf-8 -*-
"""
scripts/verify_milvus_sync.py — Milvus 数据同步校验脚本（只读）

核对「MySQL 数据源」与「Milvus 向量库」的一致性，用于数据治理巡检：
  1. 每个知识库：MySQL 已索引文档数（期望有向量）vs Milvus 实际 kb_id 向量数
  2. 孤儿向量：Milvus 中存在但 MySQL 中已无对应知识库的 kb_id 向量
  3. FAQ 行：MySQL finance_faq 条数 vs Milvus 中非知识库向量（question 维度）估算

用法：
  python scripts/verify_milvus_sync.py
  # 发现不一致时退出码为 1，可用于定时巡检 / CI 门禁
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from config import get_config  # noqa: E402
from db.milvus import get_milvus_client, ensure_collection, COLLECTION_NAME  # noqa: E402
from db.mysql import get_db  # noqa: E402


def _count_data_lines(subdir: str) -> int:
    """统计 data/<subdir>/*.jsonl 的有效行数（FAQ 数据源下限）。目录不存在返回 0。"""
    d = Path(__file__).resolve().parents[1] / "data" / subdir
    total = 0
    if d.is_dir():
        for f in d.glob("*.jsonl"):
            try:
                total += sum(1 for line in f.open(encoding="utf-8", errors="ignore")
                             if line.strip())
            except OSError:
                continue
    return total


def main() -> int:
    issues: list[str] = []
    cfg = get_config()
    client = get_milvus_client()
    ensure_collection(client)

    # ── 1. MySQL 侧数据源 ──────────────────────────────────────────────
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT k.id, k.name, "
            "(SELECT COUNT(*) FROM documents d WHERE d.kb_id = k.id AND d.status = 'indexed') AS indexed_docs "
            "FROM knowledge_bases k ORDER BY k.id"
        )
        kbs = cur.fetchall()
        cur.execute("SELECT COUNT(*) AS total FROM finance_faq")
        mysql_faq_count = cur.fetchone()["total"]
        cur.execute(
            "SELECT kb_id, COUNT(*) AS n FROM documents WHERE status = 'indexed' GROUP BY kb_id"
        )
        indexed_by_kb = {r["kb_id"]: r["n"] for r in cur.fetchall()}

    # ── 2. Milvus 侧向量统计 ───────────────────────────────────────────
    stats = client.get_collection_stats(COLLECTION_NAME)
    milvus_total = int(stats.get("row_count", 0))

    milvus_by_kb: dict[int, int] = {}
    for kb in kbs:
        try:
            milvus_by_kb[kb["id"]] = count_chunks_by_kb(kb["id"])
        except Exception as e:  # noqa: BLE001
            issues.append(f"知识库 {kb['id']} 向量计数失败: {e}")

    milvus_kb_total = sum(milvus_by_kb.values())
    milvus_faq_est = milvus_total - milvus_kb_total  # 非知识库向量（FAQ，多来源语料）

    # ── 3. 一致性比对 ──────────────────────────────────────────────────
    print(f"{'kb':>4} | {'名称':<22} | {'MySQL已索引文档':>12} | {'Milvus向量':>10} | 结论")
    print("-" * 74)
    for kb in kbs:
        kb_id, name = kb["id"], kb["name"]
        mysql_docs = indexed_by_kb.get(kb_id, 0)
        mv = milvus_by_kb.get(kb_id, 0)
        status = "OK"
        if mysql_docs > 0 and mv == 0:
            status = "⚠️ 缺失同步"
            issues.append(f"知识库 {kb_id}({name})：MySQL 有 {mysql_docs} 篇已索引文档，但 Milvus 向量为 0")
        elif mysql_docs == 0 and mv > 0:
            status = "⚠️ 孤儿向量"
            issues.append(f"知识库 {kb_id}({name})：MySQL 无已索引文档，但 Milvus 残留 {mv} 条向量")
        print(f"{kb_id:>4} | {name:<22} | {mysql_docs:>12} | {mv:>10} | {status}")

    print("-" * 74)
    print(f"Milvus 总向量: {milvus_total}（知识库 {milvus_kb_total} + FAQ {milvus_faq_est}）")
    print(f"MySQL finance_faq 条数: {mysql_faq_count}")
    # FAQ 语料多来源：data/translated 全量 + data/crawled 全量 均应进入 Milvus，
    # 而 MySQL finance_faq 仅含 translated 采样 + crawled 全量，故 Milvus FAQ 数 > MySQL 属正常。
    # 这里做「下限校验」：Milvus FAQ 不得少于 translated+crawled 全量之和（否则说明有向量缺失）。
    src_total = _count_data_lines("translated") + _count_data_lines("crawled")
    print(f"FAQ 数据源下限（translated+crawled 全量）: {src_total}")
    if milvus_faq_est < src_total:
        issues.append(
            f"FAQ 向量缺失：Milvus 仅 {milvus_faq_est} 条，低于数据源下限 {src_total} 条"
        )
    elif milvus_faq_est > mysql_faq_count + 100:
        # 正常情形（多来源），仅提示不判定为问题
        print(f"（提示）Milvus FAQ 数多于 MySQL finance_faq，差额 {milvus_faq_est - mysql_faq_count} "
              "来自 translated 全量索引等设计内来源")

    # ── 4. 汇总 ────────────────────────────────────────────────────────
    if issues:
        print("\n发现的问题：")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        logger.warning(f"Milvus 同步校验未通过：{len(issues)} 处不一致")
        return 1
    print("\n校验通过：MySQL 与 Milvus 数据一致")
    return 0


def count_chunks_by_kb(kb_id: int) -> int:
    """查询某知识库在 Milvus 中的向量数（本地复用，避免重复依赖导入顺序问题）。"""
    client = get_milvus_client()
    res = client.query(
        collection_name=COLLECTION_NAME,
        filter=f"kb_id == {int(kb_id)}",
        output_fields=["count(*)"],
    )
    try:
        return int(res[0]["count(*)"])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
