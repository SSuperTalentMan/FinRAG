#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/compliance_service.py — 合规审查流水线编排（融合 DocAudit ReviewService）。

上传建任务 → 解析（页级 digital/ocr/vl）→ 条款抽取（三重校验+重试）→ LangGraph 多 Agent
审查 → 报告/HITL。事件推 Redis List，SSE 侧按索引轮询；断点续跑按 review_task.stage。
P1 精简：条款向量不写 Milvus（规则检索走内存 BGE-M3）；审计可复用 FinRag 治理层。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import fitz

from config import get_config
from db.redis import get_redis as _get_sync_redis
from rag_qa.multimodal import ocr_parser, parsing_router, vl_fallback
from rag_qa.multimodal.clause_extract import extract_clauses
from rag_qa.multimodal.pymupdf_extractor import extract_units, open_doc, page_text_chars, render_page_png
from rag_qa.review import db as rdb
from rag_qa.review.graph import run_review
from rag_qa.review.hitl import finalize_check, now_str
from rag_qa.review.report_builder import build_markdown, build_summary
from rag_qa.review.rule_store import RuleRetriever
from rag_qa.review.schemas import PageResult, ParseUnit
from services.llm_ext import _fast_model, chat

logger = logging.getLogger(__name__)

RE_UNSAFE_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_PIPELINE_SEM = asyncio.Semaphore(2)
_running_tasks: set[asyncio.Task] = set()


def _record_review_outcome(outcome: str) -> None:
    """审查任务结果埋点。指标失败绝不影响流水线（可观测性是旁支，不能反客为主）。"""
    try:
        from services.metrics import record_review_task

        record_review_task(outcome)
    except Exception:  # noqa: BLE001
        pass


def spawn_pipeline(svc: "ComplianceService", task_id: int) -> asyncio.Task:
    t = asyncio.create_task(svc.run_pipeline(task_id))
    _running_tasks.add(t)
    t.add_done_callback(_running_tasks.discard)
    return t


async def _llm_digest(results: list[dict], clause_pages: dict[str, str]) -> str:
    """报告执行摘要：LLM 基于逐条款结果生成解读（只允许使用输入事实），失败静默降级为无摘要。"""
    from rag_qa.review.prompts import REPORT_SYSTEM
    if not results:
        return ""
    lines = []
    for r in results:
        rules = "、".join(v.get("rule_id", "") for v in (r.get("violated_rules") or []))
        lines.append(
            f"- {r.get('clause_no')}({clause_pages.get(r.get('clause_no'), '?')}) "
            f"判定:{r.get('verdict')} 风险:{r.get('risk_level')} 违反规则:{rules or '无'} "
            f"依据:{str(r.get('evidence', ''))[:120]}"
        )
    try:
        text, _ = await chat(
            [{"role": "system", "content": REPORT_SYSTEM},
             {"role": "user", "content": "\n".join(lines)}],
            model=_fast_model(), stage="report",
        )
        return text.strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("llm digest failed（降级为无摘要）: %s", e)
        return ""


class ComplianceService:
    def __init__(self, retriever: RuleRetriever):
        self.redis = _get_sync_redis()  # FinRag 同步 Redis 客户端，异步经 to_thread
        self.retriever = retriever

    # ---------- 事件通道（Redis List，SSE 侧按索引拉取） ----------
    async def _emit(self, task_id: int, event: str, data: dict):
        payload = json.dumps({"event": event, "data": data, "ts": now_str()}, ensure_ascii=False)
        key = f"finrag:compliance:events:{task_id}"
        await asyncio.to_thread(self.redis.rpush, key, payload)
        await asyncio.to_thread(self.redis.expire, key, 6 * 3600)

    async def _set_stage(self, task_id: int, stage: str, progress: dict | None = None, error: str | None = None):
        await rdb.aexecute(
            "UPDATE docaudit_review_task SET stage = %s, progress = %s, error = %s WHERE id = %s",
            (stage, json.dumps(progress or {}, ensure_ascii=False), error or "", task_id),
        )

    async def read_events(self, task_id: int, start: int) -> tuple[list[str], int, bool]:
        key = f"finrag:compliance:events:{task_id}"
        raw = await asyncio.to_thread(self.redis.lrange, key, start, -1)
        idx = start + len(raw)
        task = await rdb.afetch_one("SELECT stage FROM docaudit_review_task WHERE id = %s", (task_id,))
        stage = task["stage"] if task else "failed"
        final = stage in ("completed", "failed")
        return raw, idx, final

    # ---------- 上传建任务 ----------
    async def create_task(self, file_bytes: bytes, doc_name: str, doc_type: str, user: str) -> int:
        cfg = get_config()
        if len(file_bytes) > cfg.multimodal.upload_max_mb * 1024 * 1024:
            raise ValueError(f"文件超过 {cfg.multimodal.upload_max_mb}MB 限制")
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        dup = await rdb.afetch_one("SELECT id FROM docaudit_document WHERE file_hash = %s", (file_hash,))
        if dup:
            raise ValueError(f"文档已存在(document_id={dup['id']})，已按 SHA256 去重")
        safe_name = RE_UNSAFE_FILENAME.sub("_", Path(doc_name).name).strip(". ") or "untitled.pdf"
        uploads = Path(cfg.multimodal.uploads_dir)
        uploads.mkdir(parents=True, exist_ok=True)
        fpath = uploads / f"{file_hash[:16]}_{safe_name}"
        if fpath.resolve().parent != uploads.resolve():
            raise ValueError("非法文件路径")
        fpath.write_bytes(file_bytes)
        doc_id = await rdb.ainsert_returning_id(
            "INSERT INTO docaudit_document (doc_name, doc_type, file_path, file_hash, created_by) "
            "VALUES (%s, %s, %s, %s, %s)",
            (doc_name[:256], doc_type, str(fpath), file_hash, user[:64]),
        )
        task_id = await rdb.ainsert_returning_id(
            "INSERT INTO docaudit_review_task (document_id, stage) VALUES (%s, 'uploaded')", (doc_id,))
        await asyncio.to_thread(self.redis.delete, f"finrag:compliance:events:{task_id}")
        return task_id

    # ---------- 主流水线（后台执行，全局并发受限） ----------
    async def run_pipeline(self, task_id: int) -> None:
        # 超时保护：LLM/OCR 挂起时若不限时，任务会永久停在中间态（stage 再无流转）。
        # 超时后按 failed 收敛并推送 error/done 事件，保证前端轮询能拿到终态。
        timeout = get_config().compliance.pipeline_timeout_seconds
        async with _PIPELINE_SEM:
            try:
                if timeout and timeout > 0:
                    await asyncio.wait_for(self._pipeline(task_id), timeout=timeout)
                else:
                    await self._pipeline(task_id)
            except asyncio.TimeoutError:
                logger.error("pipeline timeout task=%s (%ss)", task_id, timeout)
                _record_review_outcome("timeout")
                await self._set_stage(task_id, "failed", error=f"流水线超时({timeout}s)")
                await self._emit(task_id, "error", {"msg": f"审查超时（超过 {timeout}s），请重试或拆分文档"})
                await self._emit(task_id, "done", {"status": "failed"})
            except asyncio.CancelledError:
                _record_review_outcome("failed")
                await self._set_stage(task_id, "failed", error="任务已取消")
                raise
            except Exception as e:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc()[-1200:]
                logger.exception("pipeline failed task={}", task_id)
                _record_review_outcome("failed")
                await self._set_stage(task_id, "failed", error=str(e)[:800])
                await self._emit(task_id, "error", {"msg": str(e)[:300], "traceback": tb})
                await self._emit(task_id, "done", {"status": "failed"})
            else:
                # 成功路径：按最终 stage 记录（completed / hitl_pending）
                _record_review_outcome(await self._final_stage(task_id))
            finally:
                # 页图是 OCR/VL 的中间产物：一页 PNG 可达数 MB，一个上百页的 PDF
                # 能留下几百 MB 垃圾。文本已入库 docaudit_page_unit、报告是 Markdown，
                # 不再需要位图，无论成功失败都清掉（失败重跑时会重新渲染）。
                await self._cleanup_page_images(task_id)

    async def _final_stage(self, task_id: int) -> str:
        try:
            row = await rdb.afetch_one(
                "SELECT stage FROM docaudit_review_task WHERE id = %s", (task_id,))
            return (row or {}).get("stage") or "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    async def _cleanup_page_images(self, task_id: int) -> None:
        """删除 uploads/pages/{doc_id} 下的中间页图。"""
        try:
            task = await rdb.afetch_one(
                "SELECT document_id FROM docaudit_review_task WHERE id = %s", (task_id,))
            if not task:
                return
            pages_dir = Path(get_config().multimodal.uploads_dir) / "pages" / str(task["document_id"])
            if not pages_dir.exists() or not pages_dir.is_dir():
                return
            removed = 0
            for p in pages_dir.iterdir():
                if p.is_file():
                    p.unlink()
                    removed += 1
            pages_dir.rmdir()
            if removed:
                logger.info("已清理中间页图 task=%s 共 %s 张", task_id, removed)
        except Exception as e:  # noqa: BLE001
            logger.warning("页图清理失败 task=%s: %s", task_id, str(e)[:120])

    async def _pipeline(self, task_id: int):
        cfg = get_config()
        task = await rdb.afetch_one("SELECT document_id, stage FROM docaudit_review_task WHERE id = %s", (task_id,))
        doc = await rdb.afetch_one(
            "SELECT id, doc_name, doc_type, file_path FROM docaudit_document WHERE id = %s", (task["document_id"],))
        file_path = doc["file_path"]
        doc_id = doc["id"]

        prev_stage = task["stage"]
        if prev_stage == "hitl_pending":
            rows = await rdb.afetch_all("SELECT summary FROM docaudit_report WHERE task_id = %s", (task_id,))
            summary = json.loads(rows[0]["summary"]) if rows and isinstance(rows[0]["summary"], str) else {}
            await self._emit(task_id, "done", {"status": "hitl_pending", "summary": summary, "resumed": True})
            return

        # ============ 阶段一：解析 ============
        await self._set_stage(task_id, "parsing")
        await self._emit(task_id, "stage", {"stage": "parsing", "msg": "开始解析"})
        page_results = await self._parse_document(doc_id, file_path, task_id)
        page_texts = {p.page_no: p.text for p in page_results}
        parser_summary: dict[str, int] = {}
        for p in page_results:
            key = p.parser_source if p.route == "digital" else f"{p.parser_source}"
            parser_summary[key] = parser_summary.get(key, 0) + 1
        await rdb.aexecute(
            "UPDATE docaudit_document SET page_count = %s, parser_summary = %s WHERE id = %s",
            (len(page_results), json.dumps(parser_summary, ensure_ascii=False), doc_id),
        )
        await self._emit(task_id, "stage", {"stage": "parsing_done", "pages": len(page_results),
                                            "parser_summary": parser_summary})

        # ============ 阶段二：条款抽取 ============
        await self._set_stage(task_id, "extracting")
        await self._emit(task_id, "stage", {"stage": "extracting", "msg": "条款结构化抽取中"})
        all_units = [u for p in page_results for u in p.units]
        clauses, defects = await extract_clauses(all_units, page_texts)
        await self._save_clauses(doc_id, clauses)
        await self._emit(task_id, "stage", {"stage": "extracting_done", "clauses": len(clauses),
                                            "defects": defects[:5]})

        # ============ 阶段三：审查（LangGraph） ============
        await self._set_stage(task_id, "reviewing")
        await self._emit(task_id, "stage", {"stage": "reviewing", "msg": "多Agent审查中"})
        clause_rows = await rdb.afetch_all(
            "SELECT id, clause_no, title, content, page_start, page_end FROM docaudit_clause "
            "WHERE document_id = %s ORDER BY id", (doc_id,))
        graph_clauses = [{"clause_id": r["id"], "clause_no": r["clause_no"],
                          "title": r["title"] or "", "content": r["content"],
                          "page_start": r["page_start"], "page_end": r["page_end"]} for r in clause_rows]
        results: list[dict] = []
        async for node, state in run_review(graph_clauses, self.retriever):
            if node == "review_batch" and isinstance(state, dict) and state.get("progress"):
                await self._emit(task_id, "progress", {"stage": "reviewing", **state["progress"]})
            if node == "report" and isinstance(state, dict) and state.get("results"):
                results = list(state["results"])
        if not results:
            results = await self._collect_results(task_id, graph_clauses)
        await self._save_reviews(task_id, results, graph_clauses)

        # ============ 阶段四：报告与 HITL ============
        summary = build_summary(results)
        clause_pages = {
            c["clause_no"]: f"第{c['page_start']}" + (f"-{c['page_end']}页" if c["page_end"] != c["page_start"] else "页")
            for c in graph_clauses
        }
        md = build_markdown(doc["doc_name"], summary, results, clause_pages,
                            llm_digest=await _llm_digest(results, clause_pages))
        reports_dir = Path(cfg.compliance.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"report_{task_id}.md"
        md_path.write_text(md, encoding="utf-8")
        await rdb.aexecute(
            "INSERT INTO docaudit_report (task_id, summary, md_path) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE summary = VALUES(summary), md_path = VALUES(md_path)",
            (task_id, json.dumps(summary, ensure_ascii=False), str(md_path)),
        )
        pending = summary["hitl_pending"]
        stage = "hitl_pending" if pending > 0 else "completed"
        await self._set_stage(task_id, stage, {"summary": summary})
        await self._emit(task_id, "stage", {"stage": stage, "summary": summary})
        await self._emit(task_id, "done", {"status": stage, "summary": summary})

    # ---------- 解析（逐页路由 + 并行） ----------
    async def _parse_document(self, doc_id: int, file_path: str, task_id: int) -> list[PageResult]:
        cfg = get_config()
        doc = open_doc(file_path)
        sem = asyncio.Semaphore(cfg.multimodal.page_concurrency)
        pages_dir = Path(cfg.multimodal.uploads_dir) / "pages" / str(doc_id)

        async def parse_one(page_no: int, page: fitz.Page) -> PageResult:
            async with sem:
                chars = page_text_chars(page)
                route = parsing_router.route_page(chars)
                try:
                    if route == "digital":
                        units = extract_units(page, page_no)
                        return PageResult(page_no=page_no, route="digital", units=units,
                                          text="\n".join(u.text for u in units),
                                          confidence=1.0, parser_source="pymupdf")
                    png = render_page_png(page)
                    img_path = ocr_parser.save_page_image(png, pages_dir, f"page_{page_no}.png")
                    units, conf = await ocr_parser.parse_page(str(img_path), page_no)
                    parser_source = "rapidocr"
                    if parsing_router.route_ocr_confidence(conf):
                        await self._emit(task_id, "progress", {
                            "stage": "parsing", "note": f"第{page_no}页置信度 {conf:.2f} → qwen-vl 兜底"})
                        units = await vl_fallback.transcribe_page(png, page_no)
                        parser_source = "qwen_vl"
                    return PageResult(page_no=page_no, route="ocr", units=units,
                                      text="\n".join(u.text for u in units),
                                      confidence=round(conf, 4), parser_source=parser_source)
                except Exception as e:  # noqa: BLE001
                    logger.warning("parse page %s failed: %s", page_no, e)
                    return PageResult(page_no=page_no, route=route, text="",
                                      parser_source="pymupdf", error=str(e)[:300])

        tasks = [parse_one(p.number + 1, p) for p in doc]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            doc.close()
        results.sort(key=lambda r: r.page_no)
        await rdb.aexecute(
            "INSERT INTO docaudit_page_unit (document_id, page_no, parser_source, route, ocr_confidence, text) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(doc_id, pr.page_no, pr.parser_source, pr.route,
              pr.confidence if pr.route == "ocr" else None, pr.text[:60000]) for pr in results],
        )
        return results

    # ---------- 条款 / 审查结果落库 ----------
    async def _save_clauses(self, doc_id: int, clauses: list[dict]):
        for c in clauses:
            await rdb.aexecute(
                "INSERT INTO docaudit_clause (document_id, clause_no, title, content, page_start, page_end, "
                "extract_status, defects) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE title=VALUES(title), content=VALUES(content), "
                "page_start=VALUES(page_start), page_end=VALUES(page_end), "
                "extract_status=VALUES(extract_status), defects=VALUES(defects)",
                (doc_id, c["clause_no"][:64], c.get("title", ""), c["content"],
                 c["page_start"], c["page_end"], c.get("extract_status", "ok"), c.get("defects", "")),
            )

    async def _save_reviews(self, task_id: int, results: list[dict], graph_clauses: list[dict]):
        for r in results:
            r.pop("_debug", None)
            await rdb.aexecute(
                "INSERT INTO docaudit_clause_review (task_id, clause_id, clause_no, verdict, risk_level, "
                "violated_rules, evidence, suggestion, hitl_status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE verdict=VALUES(verdict), risk_level=VALUES(risk_level), "
                "violated_rules=VALUES(violated_rules), evidence=VALUES(evidence), "
                "suggestion=VALUES(suggestion), hitl_status=VALUES(hitl_status), "
                "reviewer=NULL, reviewed_at=NULL",
                (task_id, r["clause_id"], r["clause_no"][:64], r["verdict"], r["risk_level"],
                 json.dumps(r.get("violated_rules", []), ensure_ascii=False),
                 r.get("evidence", ""), r.get("suggestion", ""), r.get("hitl_status")),
            )

    async def _collect_results(self, task_id: int, graph_clauses: list[dict]) -> list[dict]:
        rows = await rdb.afetch_all(
            "SELECT clause_no, clause_id, verdict, risk_level, violated_rules, evidence, suggestion, hitl_status "
            "FROM docaudit_clause_review WHERE task_id = %s", (task_id,))
        results = []
        for r in rows:
            vr = json.loads(r["violated_rules"]) if isinstance(r["violated_rules"], str) else (r["violated_rules"] or [])
            results.append({**r, "violated_rules": vr})
        return results

    # ---------- HITL 复核 ----------
    async def apply_review(self, review_id: int, action: str, reviewer: str) -> bool:
        """人工复核：approve/reject。reject 回到 pending_review 待重审。

        复核完成后做一次办结收敛：全部条款 pending 清零则任务转 completed，
        否则任务会永久停在 hitl_pending（融合前遗留问题：finalize_check 定义了却没人调）。
        """
        from rag_qa.review.hitl import can_transition
        if action not in ("approved", "rejected"):
            raise ValueError("action 仅支持 approved / rejected")
        row = await rdb.afetch_one(
            "SELECT task_id, hitl_status FROM docaudit_clause_review WHERE id = %s", (review_id,))
        if not row:
            return False
        task_id = row["task_id"]
        current = row["hitl_status"]
        target = "pending_review" if action == "rejected" else "approved"
        if not can_transition(current, target):
            raise ValueError(f"非法状态迁移: {current} -> {target}")
        await rdb.aexecute(
            "UPDATE docaudit_clause_review SET hitl_status = %s, reviewer = %s, reviewed_at = NOW() "
            "WHERE id = %s", (target, reviewer[:64], review_id))
        await self._try_finalize(task_id)
        return True

    async def _try_finalize(self, task_id: int) -> None:
        """HITL 办结收敛：pending 清零 → stage=completed 并发 done 事件。"""
        try:
            rows = await rdb.afetch_all(
                "SELECT hitl_status FROM docaudit_clause_review WHERE task_id = %s", (task_id,))
            if rows and finalize_check(rows):
                await self._set_stage(task_id, "completed")
                await self._emit(task_id, "done", {"status": "completed"})
        except Exception as e:  # noqa: BLE001
            logger.warning("finalize failed task=%s: %s", task_id, e)


compliance_service: ComplianceService | None = None


def init_service(retriever: RuleRetriever) -> ComplianceService:
    global compliance_service
    compliance_service = ComplianceService(retriever)
    return compliance_service