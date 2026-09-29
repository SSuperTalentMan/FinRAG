#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/orchestrator/nodes.py — AnswerGraph 各 Skill 节点。

节点均为 async，阻塞的 CPU/同步 IO（分类/检索/精排/LLM）用 asyncio.to_thread 包一层，
避免在事件循环内卡住其他并发请求。RAG 复用 routers.chat 的检索精排，NL2SQL 复用 P2 服务。
"""
from __future__ import annotations

import asyncio
import logging
import time

from config import get_config
from rag_qa.core.query_classifier import classify_intent
from rag_qa.orchestrator.state import AnswerState

logger = logging.getLogger(__name__)

# 复用老链路已调通的检索/降级/历史逻辑，避免重复实现且保证行为一致
from routers.chat import (
    _build_context,
    _extract_fallback_from_context,
    _load_recent_history,
)
from services.llm import generate_answer


# ─── 惰性单例：RAGSystem / NL2SQLService 复用全局，避免重复加载模型 ─────────────
_rag = None


def _get_rag():
    global _rag
    if _rag is None:
        from rag_qa.core.rag_system import RAGSystem

        _rag = RAGSystem()
    return _rag


_nl2sql_svc = None


def _get_nl2sql():
    """问数服务单例。

    MetaStore 与路由层共用同一实例（rag_qa.orchestrator.route.get_route_meta），
    避免路由的实体词表与问数的 Schema 召回各加载一份元数据。
    """
    global _nl2sql_svc
    if _nl2sql_svc is None:
        from rag_qa.nl2sql.service import Nl2SqlService
        from rag_qa.orchestrator.route import get_route_meta

        _nl2sql_svc = Nl2SqlService(get_route_meta())
    return _nl2sql_svc


# ─── 各 Skill 节点 ──────────────────────────────────────────────────────────────
async def rag_retrieve_node(state: AnswerState) -> AnswerState:
    """RAG：意图分域 -> 混合检索 -> 精排，装 context/sources。"""
    question = state["question"]
    top_k = get_config().orchestrator.answer_top_k

    intent = await asyncio.to_thread(classify_intent, question)
    domain = state.get("domain") or intent.domain
    state["domain"] = domain
    state["trace"].append(f"rag.domain={domain}")

    # _build_context 阻塞（Milvus/BM25/Reranker），线程池执行
    context, sources = await asyncio.to_thread(
        _build_context, question, intent, None
    )
    state["context"] = context
    state["sources"] = sources
    state["stage"] = "rag_retrieve"
    return state


async def rag_answer_node(state: AnswerState) -> AnswerState:
    """RAG：组装最终回答。检索空结果时防幻觉直返提示；LLM 失败降级取首条 QA。"""
    question = state["question"]
    context = state.get("context", "")
    domain = state.get("domain", "")

    if not context:
        state["answer"] = "抱歉，未在知识库中找到与您问题相关的资料。请尝试换种问法或联系人工客服。"
        state["status"] = "ok"
        state["stage"] = "rag_answer"
        return state

    history = ""
    if state.get("session_id"):
        history = await asyncio.to_thread(
            _load_recent_history, state["session_id"], state.get("user_id")
        )
    try:
        answer = await asyncio.to_thread(generate_answer, question, context, domain, history)
        state["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        answer = _extract_fallback_from_context(context)
        state["status"] = "degraded"
        state["degraded"] = True
        state["error"] = str(e)[:300]
    state["answer"] = answer
    state["stage"] = "rag_answer"
    return state


async def nl2sql_node(state: AnswerState) -> AnswerState:
    """NL2SQL：整条 ChatBI 问数算子（意图->Schema召回->SQL生成->SQLGuard->只读执行）。

    注意：LLM 配额由入口 routers.chat.ask_unified / ask_unified_stream 统一扣减，
    此处不再重复检查（否则一次问数扣两份配额）。
    """
    question = state["question"]

    svc = _get_nl2sql()
    await svc.ensure_ready()
    ans = await svc.ask(question, role=state.get("role", "user"))

    # 可观测：融合进来就一定要能量化，否则无法判断问数到底好不好用
    try:
        from services.metrics import record_nl2sql

        outcome = "cached" if getattr(ans, "cached", False) else (
            "data" if ans.answer_type == "data" else (ans.status or "failed"))
        record_nl2sql(outcome, (ans.latency_ms or 0) / 1000.0)
    except Exception:  # noqa: BLE001  指标失败绝不能影响主流程
        pass

    if ans.answer_type == "data":
        state["answer"] = ans.text
        state["sql"] = ans.sql or ""
        state["table"] = {
            "columns": ans.columns,
            "rows": ans.rows,
            "row_count": ans.row_count,
            "tables": ans.tables,
        }
        state["meta"] = {
            "assumptions": ans.assumptions,
            "repair_rounds": ans.repair_rounds,
            "latency_ms": ans.latency_ms,
        }
        state["status"] = "ok"
    else:
        state["answer"] = ans.text
        state["status"] = "ok" if ans.status == "ok" else (
            "refused" if ans.status in ("refused", "clarify") else "error"
        )
    state["stage"] = "nl2sql"
    state["trace"].append(f"nl2sql.{ans.answer_type}")
    return state


async def multimodal_node(state: AnswerState) -> AnswerState:
    """多模态：识别/读图类问题走不了纯文本 RAG，直接给出上传引导。

    融合前 multimodal 被静默归并到 rag 分支，用户问"这张图里写着什么"会被当成
    文档问答去检索知识库，返回一堆无关条文。这里显式接管：说明能力边界并指向
    /review/upload（DocAudit 的解析+审查流水线），不编造答案。
    """
    hint = (
        "我识别到你的问题涉及图片/扫描件内容。对话窗口暂不能直接读取图片，"
        "请通过「合规审查」上传 PDF 或图片，系统会自动完成版面解析（数字文本/OCR/视觉兜底）、"
        "条款抽取与合规审查，并生成可下载的审查报告。上传后可在这里追问报告结论。"
    )
    state["answer"] = hint
    state["status"] = "ok"
    state["stage"] = "multimodal"
    state["trace"].append("multimodal.upload_guide")
    return state


async def review_node(state: AnswerState) -> AnswerState:
    """文档审查（融合 DocAudit）：对话内直接受理，不必跳去上传页。

    两种形态：
    - 问题里带了待审正文（粘贴的合同条款等）→ 合成 PDF 直接建任务并后台跑流水线，
      回 task_id 让用户可在此追问进展；
    - 只是问"能审吗/帮我审这份文件"→ 给出上传引导。

    为什么合成 PDF 而不是把文本直接喂给审查图：DocAudit 的条款抽取依赖
    页码回验（coverage_threshold），必须有真实分页；用 fitz 按页写入即可
    复用整条解析→抽取→审查→报告链路，零改造。
    """
    from rag_qa.orchestrator.review_bridge import (
        accepted_inline_text,
        build_review_guide,
        submit_inline_review,
    )

    question = state["question"]
    cfg = get_config().orchestrator
    body = accepted_inline_text(question)

    if not body:
        state["answer"] = build_review_guide()
        state["status"] = "ok"
        state["stage"] = "review"
        state["trace"].append("review.guide")
        return state

    if len(body) > cfg.review_inline_max_chars:
        state["answer"] = (
            f"你粘贴的内容约 {len(body)} 字，超过了对话内直接审查的上限"
            f"（{cfg.review_inline_max_chars} 字）。请通过「合规审查」上传原文件，"
            "系统会保留原始版面与页码，审查结论可溯源。"
        )
        state["status"] = "refused"
        state["stage"] = "review"
        state["trace"].append("review.too_long")
        return state

    try:
        task_id, doc_name = await submit_inline_review(
            body, user=state.get("username", "") or f"user_{state.get('user_id') or 0}"
        )
        try:
            from services.metrics import record_review_task

            record_review_task("accepted")
        except Exception:  # noqa: BLE001
            pass
        state["task"] = {"task_id": task_id, "status": "uploaded", "doc_name": doc_name}
        state["answer"] = (
            f"已受理，审查任务 #{task_id} 已启动（来源：{doc_name}）。\n"
            "流水线会依次完成版面解析 → 条款抽取 → 逐条合规审查 → 生成报告；"
            "命中低置信条款会转人工复核。稍等片刻可以在这里问我「任务进度如何」，"
            "或在合规审查页查看完整报告。"
        )
        state["status"] = "ok"
        state["trace"].append(f"review.task={task_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning("对话内审查受理失败: %s", str(e)[:200])
        state["answer"] = (
            "审查服务暂时不可用，请稍后重试，或通过「合规审查」页面上传文件。"
            f"（{_brief_err(e)}）"
        )
        state["status"] = "degraded"
        state["degraded"] = True
        state["error"] = str(e)[:300]
        state["trace"].append("review.failed")
    state["stage"] = "review"
    return state


def _brief_err(e: Exception) -> str:
    s = str(e)
    return (s[:120] + "…") if len(s) > 120 else s


async def chitchat_node(state: AnswerState) -> AnswerState:
    """寒暄/问候：简短通用回答，不检索知识库。"""
    question = state["question"]
    prompt = (
        "你是 FinRag 智能金融助手。用户正在问候你，请用简短友好的一句话回应，"
        "并提示你既能回答金融知识/政策文档问题，也能查询经营数据（销售额、订单、退货、排名等）"
        "以及读取扫描件/图片。不要虚构能力。"
        f"\n\n用户说：{question}"
    )
    try:
        # 轻量任务：复用主链路 chat_completion（含降级）
        from services.llm import chat_completion

        answer = await asyncio.to_thread(
            chat_completion, [{"role": "user", "content": prompt}]
        )
    except Exception as e:  # noqa: BLE001
        answer = "你好！我可以帮你查询经营数据、检索金融文档，或读取图片/扫描件，请问你想了解什么？"
        state["degraded"] = True
        state["error"] = str(e)[:200]
    state["answer"] = answer
    state["status"] = "ok"
    state["stage"] = "chitchat"
    return state