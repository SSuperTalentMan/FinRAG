#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/nl2sql/service.py — 问数编排（自然语言 → SQL Guard → 只读执行 + 自修复）。

链路：意图粗判 → Schema 召回 → SQL 生成 → SQLGuard 校验 → 只读执行 → 结果。
失败自修复：Guard 拒绝 / 执行报错 → 错误信息回喂 LLM 重写（≤ repair_max_rounds 轮）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from config import get_config
from db.redis import cache_get, cache_set
from rag_qa.nl2sql import executor, sqlguard
from rag_qa.nl2sql import caliber_check
from rag_qa.nl2sql.generation import generate_sql
from rag_qa.nl2sql.schema import Nl2SqlAnswer
from rag_qa.nl2sql.schema_retriever import SchemaRetriever
from rag_qa.nl2sql.meta_store import MetaStore

logger = logging.getLogger(__name__)

# 问数语义缓存：同问句（含角色维度）在 TTL 内复用 SQL 与结果集，省掉一次
# LLM 生成 + 一次业务库查询。key 走 db.redis 的 finrag:qa: 前缀，
# 因此文档变更触发的 clear_qa_cache() 会一并失效，不会留下跨模块的脏缓存。
# 单条缓存体积上限：结果集可能上千行，超限不写，避免撑大 Redis。
_NL2SQL_CACHE_MAX_BYTES = 256 * 1024

_GREETINGS = re.compile(
    r"^(你好|您好|嗨|哈喽|hello|hi|早|晚上好|谢谢|感谢|再见|拜拜|你是谁|你在干嘛|在吗)",
    re.IGNORECASE,
)
_BIZ_HINT = re.compile(
    r"(销售额|GMV|客单价|订单|销量|退货|退款|利润|客户|会员|门店|城市|大区|商品|"
    r"天气|GDP|CPI|数量|金额|同比|环比|趋势|多少|几个|统计|查询|榜单|排|TOP|增长率)",
    re.IGNORECASE,
)


class Nl2SqlService:
    def __init__(self, meta: MetaStore):
        self.meta = meta
        self.retriever = SchemaRetriever(meta)

    async def ensure_ready(self) -> None:
        await self.meta.ensure_loaded()

    @staticmethod
    def is_chitchat(question: str) -> bool:
        q = question.strip()
        return bool(_GREETINGS.match(q)) and not _BIZ_HINT.search(q)

    def _build_data_answer(self, question, sql, columns, rows, tables, rounds, latency, assumptions) -> Nl2SqlAnswer:
        text = _format_result(question, columns, rows, len(tables) > 0)
        return Nl2SqlAnswer(
            answer_type="data", status="ok", sql=sql, assumptions=assumptions,
            columns=columns, rows=rows, row_count=len(rows), tables=tables,
            repair_rounds=rounds, latency_ms=latency, text=text,
        )

    async def ask(self, question: str, role: str = "user") -> Nl2SqlAnswer:
        """端到端问数，返回结构化答案。

        命中语义缓存时直接回缓存（标记 cached=True），否则走完整链路并把成功结果写缓存。
        缓存 key 含 role：不同角色的表级白名单不同，混用会越权。
        """
        t0 = time.monotonic()
        cfg = get_config().nl2sql
        if self.is_chitchat(question):
            return Nl2SqlAnswer(
                answer_type="chitchat", status="ok",
                text="你好！我可以帮你查询经营数据（销售额、订单、退货、客户、门店等）。"
                     "请告诉我你想查什么。", latency_ms=int((time.monotonic() - t0) * 1000),
            )

        cache_key = _ask_cache_key(question, role)
        if cfg.cache_ttl_seconds > 0:
            hit = _cache_read(cache_key)
            if hit is not None:
                hit.cached = True
                hit.latency_ms = int((time.monotonic() - t0) * 1000)
                return hit

        ans = await self._ask_uncached(question, role)
        if cfg.cache_ttl_seconds > 0 and ans.answer_type == "data" and ans.status == "ok":
            _cache_write(cache_key, ans, cfg.cache_ttl_seconds)
        return ans

    async def _ask_uncached(self, question: str, role: str) -> Nl2SqlAnswer:
        """真正跑一遍：Schema 召回 → 生成 → Guard → 只读执行（含自修复）。"""
        t0 = time.monotonic()
        cfg = get_config().nl2sql

        ctx = await self.retriever.retrieve(question, role)
        allowed = self.retriever.allowed_tables_for(role)
        if not ctx.ddl_texts:
            return Nl2SqlAnswer(
                answer_type="refused", status="refused",
                text='没有在业务库中找到与问题相关的表结构，无法查询。'
                     '麻烦换一种问法，例如「近7天各区域销售额是多少」。',
                tables=[], latency_ms=int((time.monotonic() - t0) * 1000),
            )

        failure: str = ""
        last_round = cfg.repair_max_rounds
        total_budget = cfg.gen_timeout_seconds * 2  # 总预算：单轮超时×2，杜绝 80s+ 空转
        for attempt in range(cfg.repair_max_rounds + 1):
            last_round = attempt
            # 总预算保护：多轮修复累计超时即拒答收尾，避免把延迟堆到 80s+
            if time.monotonic() - t0 > total_budget:
                return Nl2SqlAnswer(
                    answer_type="refused", status="refused",
                    text="生成查询超时（多次修复未通过），请简化问题或确认业务库覆盖该指标。",
                    tables=list(ctx.table_names), repair_rounds=attempt,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            # 1) SQL 生成（带硬超时，防止模型挂起把延迟堆到 38s+）
            try:
                gen, _, _ = await asyncio.wait_for(
                    generate_sql(ctx, failure_context=failure),
                    timeout=cfg.gen_timeout_seconds,
                )
            except asyncio.TimeoutError:
                failure = "SQL 生成超时（模型响应过慢），请简化问题或稍后重试"
                logger.warning("nl2sql generate timeout (round %s, %ss)", attempt, cfg.gen_timeout_seconds)
                continue
            except Exception as e:  # noqa: BLE001
                failure = f"SQL 生成失败: {e}"
                logger.warning("nl2sql generate failed (round %s): %s", attempt, e)
                continue
            # 后处理：剥离模型误塞的注释/推理文本（--、/* */、//），避免 SQL 不可解析
            sql = _strip_sql_comments(gen.sql or "")
            if not sql:
                if (gen.assumptions or "").strip().upper().startswith("REFUSE"):
                    return Nl2SqlAnswer(
                        answer_type="refused", status="refused",
                        text=gen.assumptions or "业务库无此数据", assumptions=gen.assumptions,
                        tables=list(ctx.table_names), repair_rounds=attempt,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                # 剥离后无 SQL（模型只输出了推理未给语句）：已修复一轮则干净拒答收尾，避免空转
                if attempt >= 1:
                    return Nl2SqlAnswer(
                        answer_type="refused", status="refused",
                        text="未能从可用表结构中拼出可执行的查询（缺少必要的关联路径）。",
                        tables=list(ctx.table_names), repair_rounds=attempt,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                failure = "请直接输出纯 SELECT 语句，禁止在 sql 中写注释或解释性文字"
                continue
            # 派生指标口径确定性回检：命中明显偏口径写法（客单价/退货率/季度/同比环比）
            # 则回喂模型重写，避免返回"能执行但口径错"的结果。正确 SQL 不受影响。
            caliber_warn = caliber_check.check_critical_caliber(gen, question)
            if caliber_warn:
                if attempt >= cfg.repair_max_rounds:
                    # 已无重写机会：仍返回数据（口径风险由评估判分暴露），但记录告警
                    logger.warning("nl2sql 口径回检未收敛（放弃重写）: %s | sql=%s",
                                   caliber_warn, sql[:160])
                else:
                    failure = f"口径校验未通过: {caliber_warn}；请严格参照派生指标口径示例重写"
                    continue
            # 2) SQLGuard 校验
            gr = sqlguard.check(sql, allowed)
            if not gr.ok:
                reason = gr.reject_reason
                # 注释/语法类污染：已修复一轮仍拼不出合法 SQL → 直接拒答收尾，不再空转到第 3 轮
                # （这类问题通常缺关联路径，重试无意义，且会把延迟堆到 80s+）
                if ("解析失败" in reason or "Unexpected token" in reason) and attempt >= 1:
                    return Nl2SqlAnswer(
                        answer_type="refused", status="refused",
                        text="未能生成语法合法的查询（可用表结构不足以支持该问法）。",
                        tables=list(ctx.table_names), repair_rounds=attempt,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                failure = f"SQL 未通过安全校验: {reason}"
                # 语法/解析类失败：回喂定向修复提示，提升后续轮次收敛率（根治漏写 JOIN/别名/注释）
                if "解析失败" in reason or "Unexpected token" in reason:
                    failure += ("；请只输出纯 SQL，禁止任何注释(--/#/* */)与解释文字；"
                                "多表关联必须用显式 JOIN ... ON，禁止逗号隐式连接；"
                                "所有表别名须先在 FROM/JOIN 中定义且全程一致；不要遗漏关键字。")
                logger.warning("nl2sql guard reject (round %s): %s", attempt, reason)
                continue
            # 3) 只读执行（白名单随 SQL 传入，执行器内部二次校验，纵深防御）
            try:
                columns, rows = await executor.run_readonly_sql(gr.normalized_sql, allowed)
            except executor.GuardReject as e:
                failure = e.reason
                continue
            except Exception as e:  # noqa: BLE001
                failure = f"SQL 执行失败: {e}"
                continue
            return self._build_data_answer(
                question, gr.normalized_sql, columns, rows, gr.tables, attempt,
                int((time.monotonic() - t0) * 1000), gen.assumptions,
            )

        # 自修复耗尽：略带脱敏的失败原因（不外泄内部堆栈）
        status = "guard_rejected" if "安全校验" in failure or "白名单" in failure else "failed"
        return Nl2SqlAnswer(
            answer_type="error", status=status, text=f"未能生成可执行的查询：{_brief(failure)}",
            tables=list(ctx.table_names), repair_rounds=last_round,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    async def execute_direct(self, sql: str, role: str = "user") -> Nl2SqlAnswer:
        """直接粘贴 SQL 的问数路径：仍走 SQLGuard 校验 + 只读执行（不信任外部 SQL）。"""
        t0 = time.monotonic()
        await self.meta.ensure_loaded()
        allowed = self.retriever.allowed_tables_for(role)
        gr = sqlguard.check(sql, allowed)
        if not gr.ok:
            return Nl2SqlAnswer(
                answer_type="error", status="guard_rejected", text=f"SQL 被安全层拒绝：{gr.reject_reason}",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        try:
            columns, rows = await executor.run_readonly_sql(gr.normalized_sql, allowed)
        except executor.GuardReject as e:
            return Nl2SqlAnswer(answer_type="error", status="guard_rejected", text=e.reason,
                                latency_ms=int((time.monotonic() - t0) * 1000))
        except Exception as e:  # noqa: BLE001
            return Nl2SqlAnswer(answer_type="error", status="failed", text=f"执行失败：{_brief(e)}",
                                latency_ms=int((time.monotonic() - t0) * 1000))
        return self._build_data_answer(
            "直接 SQL", gr.normalized_sql, columns, rows, gr.tables, 0,
            int((time.monotonic() - t0) * 1000), "",
        )


def _ask_cache_key(question: str, role: str) -> str:
    """问数缓存 key。

    角色必须进 key：allowed_tables_for(role) 决定可见表，admin 与普通用户的
    结果集不同，混 key 会让低权限用户读到高权限表的数据。
    """
    norm = " ".join((question or "").split()).lower()
    return f"nl2sql:{role or 'user'}:{norm}"


def _cache_read(key: str) -> Nl2SqlAnswer | None:
    try:
        raw = cache_get(key)
        if not isinstance(raw, dict):
            return None
        return Nl2SqlAnswer(**raw)
    except Exception as e:  # noqa: BLE001
        logger.debug("问数缓存读取失败（按未命中处理）: %s", str(e)[:120])
        return None


def _cache_write(key: str, ans: Nl2SqlAnswer, ttl: int) -> None:
    try:
        payload = ans.model_dump()
        payload["cached"] = False  # 缓存里存的是原始结果，命中时才置 True
        if len(json.dumps(payload, ensure_ascii=False, default=str)) > _NL2SQL_CACHE_MAX_BYTES:
            logger.debug("问数结果过大，跳过缓存: %s", key[:48])
            return
        cache_set(key, payload, ttl=ttl)
    except Exception as e:  # noqa: BLE001
        logger.debug("问数缓存写入失败（忽略）: %s", str(e)[:120])


def _brief(msg) -> str:
    """收敛异常文本，避免把数据库内部堆栈抛给客户端。"""
    s = str(msg)
    return (s[:180] + "…") if len(s) > 180 else s


def _strip_sql_comments(sql: str) -> str:
    """剥离模型误塞进 sql 字段的注释：行注释 --、块注释 /* */、C++ 风格 //。

    部分模型会在 JSON 的 sql 值里夹带推理（如 '-- fact_order_items 没有 customer_id'、
    '// 结论：无法构建 JOIN'），导致 SQL 不可解析。剥离后若剩余为空说明模型只输出了
    思考未给出可执行语句，调用方据此拒答而非空转重试。
    """
    if not sql:
        return ""
    # 块注释（可能跨行）
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # 行注释 -- 到行尾（SQL 合法值里极少出现连续双横线，可安全剥离）
    sql = re.sub(r"--[^\n]*", " ", sql)
    # C++ 风格 // 到行尾
    sql = re.sub(r"//[^\n]*", " ", sql)
    return sql.strip()


def _format_result(question: str, columns: list[str], rows: list[list], has_table: bool) -> str:
    """结果集的人性化摘要：避免把原始错误/大结果一次性倒给用户。"""
    n = len(rows)
    if n == 0:
        return "查询完成：在所选业务表中未找到满足条件的数据。"
    head = "，".join(columns[:6]) + ("…" if len(columns) > 6 else "")
    first = rows[0]
    sample = "；".join(f"{c}={first[i]}" for i, c in enumerate(columns[:4]))
    tail = f"示例：{sample}。" if sample else ""
    return f"共查到 {n} 条结果（字段：{head}）。{tail}"