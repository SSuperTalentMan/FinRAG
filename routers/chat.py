#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
routers/chat.py — 核心问答路由
实现完整 RAG 检索管线：Redis缓存 → BM25精确匹配 → 意图识别 → Milvus向量检索 → BGE-Reranker重排 → LLM生成回答
支持流式（SSE）和非流式两种响应模式。
"""

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from models import ChatRequest, ChatResponse, ApiResponse
from config import get_config
from db.redis import cache_get, cache_set, get_redis, append_session_message, SESSION_KEY_PREFIX
from rag_qa.core.query_classifier import classify_intent
from rag_qa.core.prompts import sanitize_input, RAGPrompts
from rag_qa.core.strategy_selector import (
    STRATEGY_DIRECT,
    STRATEGY_HYDE,
    STRATEGY_SUBQUERY,
    STRATEGY_BACKTRACK,
    select_strategy,
)
from services.bm25 import get_bm25_retriever
from services.embedding import encode_query_dense_sparse
from db.milvus import get_milvus_client, search_milvus, ensure_collection
from services.reranker import get_reranker, is_chunk_id, rerank, rerank_text
from services.llm import generate_answer, get_llm_client, get_async_llm_client, chat_completion
from services.metrics import record_retrieval, record_llm, record_rate_limit, record_strategy
from services.market_data import build_market_context, is_realtime_query, last_snapshot_meta
from services.rate_limiter import check_rate_limit
from routers.auth import get_current_user

router = APIRouter(prefix="/chat", tags=["问答"])


def _check_llm_quota(user_id: int) -> None:
    """用户维度 LLM 配额检查，超限抛 429。Redis 不可用时 fail-open（放行）。"""
    cfg = get_config()
    allowed, _ = check_rate_limit(
        f"user:{user_id}:llm", cfg.rate_limit.llm_quota_max, cfg.rate_limit.llm_quota_window
    )
    if not allowed:
        record_rate_limit("llm_quota")
        raise HTTPException(status_code=429, detail="已超出每小时问答配额，请稍后再试")


def _mask_question(question: str, keep: int = 10) -> str:
    """问题文本脱敏：仅保留前 N 字符用于日志定位，避免完整提问（可能含敏感信息）落盘。"""
    if not question:
        return ""
    return f"{question[:keep]}…(len={len(question)})"


# ─── 会话保存辅助 ──────────────────────────────────────────────────────────────
# 会话 key 前缀 / TTL 已收拢到 db.redis（单一事实来源），此处仅导入使用

# BM25 直接返回（早退）的分数门槛。低于该分数时，BM25 命中项仅作为候选
# 与 Milvus 文档块一起进入 Reranker 精排，不再直接返回。
BM25_EARLY_RETURN_THRESHOLD = 0.95

# 早退的第二道门控：绝对相关性下限。
# softmax_score 是在**全部 FAQ 候选**上做 softmax（见 services/bm25.py::search），
# 衡量的是"这条比其他 FAQ 领先多少"（相对分数），而非"它是否真的回答了这个问题"。
# 只要某条 FAQ 的 BM25 原始分明显领先，softmax 就会逼近 1.0，单纯调高阈值治不了本。
# 实测事故：
#   「股票发行注册制改革的主要内容是什么？」→ 误命中 FAQ「债券注册制改革全面落地」
#   softmax=0.985（≥0.95 触发早退）但 cross-encoder 相关性仅 0.047，
#   导致 doc_4（股票发行注册制投教问答）被完全挡在检索之外，答非所问。
# 因此早退必须再过一道 Reranker 绝对相关性校验。校准数据见
# scripts/_probe_bm25_gate.py：误命中负样本 rerank=0.047，
# FAQ 精确命中/口语化改写正样本 rerank=0.9988~1.0000，两者区间极宽，取 0.90 安全。
BM25_EARLY_RETURN_RERANK_MIN = 0.90


def _bm25_early_gate_passed(question: str, bm25_best: dict) -> bool:
    """BM25 早退前的绝对相关性门控。

    不达标则放弃早退，把 BM25 命中项交回完整管线，与 Milvus 文档块一起精排，
    由相关性决定最终答案。打分失败时保守放行——完整管线同样依赖 Reranker，
    此时放行等同于退回门控引入前的行为，不会让接口整体不可用。
    """
    try:
        score = float(get_reranker().predict([(question, rerank_text(bm25_best))])[0])
    except Exception as e:
        logger.warning(f"BM25 早退门控打分失败，保守放行: {e}")
        return True
    if score < BM25_EARLY_RETURN_RERANK_MIN:
        logger.info(
            f"BM25 早退被门控拦下（rerank={score:.4f} < {BM25_EARLY_RETURN_RERANK_MIN}）："
            f"FAQ「{bm25_best['question'][:40]}」与问题相关性不足，转入完整检索管线"
        )
        return False
    logger.info(f"BM25 早退门控通过（rerank={score:.4f}）")
    return True


def _save_chat_message(session_id: str, role: str, content: str, domain: str = "", user_id: int = None) -> None:
    """保存一条对话消息到 Redis 会话历史，绑定 user_id。

    底层走 Lua 原子追加（db.redis.append_session_message），并发写不丢消息，
    追加成功后同步刷新该用户的会话索引。失败仅告警（fail-open，不阻断问答）。
    """
    if not session_id:
        return
    try:
        msg = {"role": role, "content": content, "domain": domain, "ts": int(time.time())}
        append_session_message(session_id, msg, user_id=user_id)
    except Exception as e:
        logger.warning(f"保存会话消息失败: {e}")


# ─── Context token 预算控制 ─────────────────────────────────────────────────────
# 粗略 token 估算：中文约 2 字/token，英文约 4 字符/token，取折中 2.5 字符/token。
# 比精确 tokenizer 快得多，误差在 20% 以内，对截断保护已足够。
_CHARS_PER_TOKEN = 2.5
# 上下文 token 预算：为 prompt 模板、问题、历史留出空间后，context 的最大 token 数。
# qwen3.6-flash 上下文窗口 32K，max_tokens=2048，历史预算 ~1000，余下给 context。
_CONTEXT_TOKEN_BUDGET = 6000


def _estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数（无需加载 tokenizer，误差 ~20%）。"""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _truncate_context(context: str, budget: int = _CONTEXT_TOKEN_BUDGET) -> str:
    """按 token 预算截断 context，避免超出 LLM 上下文窗口。

    截断在段落边界（\\n\\n）进行，保留前 N 段不超过预算。
    """
    if not context:
        return context
    if _estimate_tokens(context) <= budget:
        return context
    # 按段落分割，逐段累加直到超预算
    paragraphs = context.split("\n\n")
    result = []
    total = 0
    for p in paragraphs:
        p_tokens = _estimate_tokens(p)
        if total + p_tokens > budget:
            break
        result.append(p)
        total += p_tokens
    logger.warning(f"Context 超预算截断: {_estimate_tokens(context)} -> {total} tokens")
    return "\n\n".join(result) if result else context[:int(budget * _CHARS_PER_TOKEN)]


# ─── 对话历史滑窗 ──────────────────────────────────────────────────────────────
_HISTORY_TURNS = 4  # 纳入 LLM 的最近对话轮数（1 轮 = 1 user + 1 assistant）
_HISTORY_TOKEN_BUDGET = 1000  # 历史占用 token 预算


def _load_recent_history(session_id: str, user_id: int) -> str:
    """从 Redis 加载最近 N 轮对话历史，格式化为文本供 prompt 使用。

    仅取最近 _HISTORY_TURNS*2 条消息（user+assistant 交替），
    并按 token 预算截断，避免历史过长挤占 context 空间。
    """
    if not session_id:
        return ""
    try:
        r = get_redis()
        key = f"{SESSION_KEY_PREFIX}{session_id}"
        raw = r.get(key)
        if not raw:
            return ""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        messages = data.get("messages", [])
        if not messages:
            return ""
        # 取最近 N 轮（每轮 = user + assistant，即 2 条）
        recent = messages[-(_HISTORY_TURNS * 2):]
        # 格式化为文本，按 token 预算截断
        lines = []
        total = 0
        for m in recent:
            role_label = "用户" if m.get("role") == "user" else "助手"
            line = f"{role_label}: {m.get('content', '')}"
            line_tokens = _estimate_tokens(line)
            if total + line_tokens > _HISTORY_TOKEN_BUDGET:
                break
            lines.append(line)
            total += line_tokens
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"加载对话历史失败（忽略历史）: {e}")
        return ""


def _extract_fallback_from_context(context: str) -> str:
    """LLM 故障降级：从检索 context 中提取首条 QA 的回答作为兜底答案。

    context 格式为「【问题】xxx（来源：yyy）\n【回答】zzz\n\n...」，
    提取第一个「【回答】」后的内容。无法提取时返回通用提示。
    """
    if not context:
        return "抱歉，回答生成服务暂时不可用，请稍后重试。"
    try:
        # 提取第一个 【回答】 后的内容（到下一个 【问题】 或结尾）
        parts = context.split("【回答】", 2)
        if len(parts) >= 2:
            answer = parts[1].split("【问题】")[0].strip()
            if answer:
                return answer
    except Exception:
        pass
    return "抱歉，回答生成服务暂时不可用，请稍后重试。已为您检索到相关资料，可查看下方来源。"


def _hybrid_recall(query: str, domain: str | None) -> list[dict]:
    """
    召回层：Milvus 混合检索（稠密+稀疏）+ BM25 候选，融合去重并回填真实来源。
    各检索策略（直接/HyDE/子查询/回溯）共用同一召回核心，保证行为一致。
    """
    client = get_milvus_client()
    ensure_collection(client)
    query_dense, query_sparse = encode_query_dense_sparse(query)
    cfg = get_config()
    milvus_domain = domain if domain and domain != "general" else None
    milvus_hits = search_milvus(client, query_dense, query_sparse, k=cfg.retrieval.retrieval_k, domain=milvus_domain)

    # 兜底：领域过滤零命中时回退全库检索一次。
    # 意图分类把问题路由到无内容的领域（如「存款保险」被判为 insurance）时，
    # 若不做回退会直接落到"无检索结果"，只能让 LLM 凭空作答。
    if not milvus_hits and milvus_domain:
        logger.warning(f"领域 '{milvus_domain}' 过滤无命中，回退全库检索")
        milvus_hits = search_milvus(client, query_dense, query_sparse, k=cfg.retrieval.retrieval_k, domain=None)

    # BM25 候选
    bm25_retriever = get_bm25_retriever()
    bm25_hits = bm25_retriever.search(query, domain=domain, top_k=5) or []

    # 融合去重（先保留检索通道标签，稍后统一回填真实来源）
    candidates: list[dict] = []
    seen: set[str] = set()
    for hit in bm25_hits:
        if hit["question"] not in seen:
            candidates.append({**hit, "source": hit.get("source") or "bm25"})
            seen.add(hit["question"])
    for hit in milvus_hits:
        if hit["question"] not in seen:
            candidates.append({
                "question": hit["question"], "answer": hit["answer"],
                # text 为子块原文，供 Reranker 使用（与向量化文本一致）
                "text": hit.get("text", ""),
                "category": hit["category"], "score": hit["score"],
                "source": hit.get("source", "milvus"),
            })
            seen.add(hit["question"])

    # 回填真实来源（source / source_url）便于答案溯源
    # Milvus 实体未存来源，按 question 回查 MySQL finance_faq
    if candidates:
        try:
            from db.mysql import get_question_source_map
            src_map = get_question_source_map()
            for c in candidates:
                s = src_map.get(c["question"], {})
                if s.get("source"):
                    c["source"] = s["source"]
                c["source_url"] = s.get("source_url", "")
        except Exception as e:
            logger.warning(f"来源回填失败（使用检索通道标签）: {e}")

    return candidates


def _recall_with_strategy(query: str, domain: str | None, strategy: str) -> list[dict]:
    """按选定策略召回。改写类策略失败时一律降级回原始查询直接召回，保证可用性。

    注意：HyDE/回溯改写只影响"用什么文本去召回"，精排仍然用用户原始问题打分，
    保证相关性始终以用户意图为准。
    """
    if strategy == STRATEGY_HYDE:
        try:
            hypo = chat_completion(
                messages=[{"role": "user", "content": RAGPrompts.hyde_prompt(query)}],
                stream=False,
            ).strip()
            if hypo:
                logger.info(f"HyDE 假设答案: {hypo[:60]}...")
                return _hybrid_recall(hypo, domain)
        except Exception as e:
            logger.warning(f"HyDE 改写失败，降级直接检索: {e}")
        return _hybrid_recall(query, domain)

    if strategy == STRATEGY_SUBQUERY:
        try:
            sub_text = chat_completion(
                messages=[{"role": "user", "content": RAGPrompts.subquery_prompt(query)}],
                stream=False,
            )
            # 上限 3 个子查询：召回成本（embedding+Milvus 检索）随子查询数线性增长
            sub_queries = [q.strip() for q in sub_text.strip().splitlines() if q.strip()][:3]
            if sub_queries:
                logger.info(f"子查询分解: {sub_queries}")
                merged: dict[str, dict] = {}
                for sq in sub_queries:
                    for c in _hybrid_recall(sq, domain):
                        q = c["question"]
                        if q not in merged or c.get("score", 0) > merged[q].get("score", 0):
                            merged[q] = c
                if merged:
                    return list(merged.values())
        except Exception as e:
            logger.warning(f"子查询分解失败，降级直接检索: {e}")
        return _hybrid_recall(query, domain)

    if strategy == STRATEGY_BACKTRACK:
        try:
            simplified = chat_completion(
                messages=[{"role": "user", "content": RAGPrompts.backtracking_prompt(query)}],
                stream=False,
            ).strip()
            if simplified:
                logger.info(f"回溯简化问题: '{simplified[:60]}'")
                return _hybrid_recall(simplified, domain)
        except Exception as e:
            logger.warning(f"回溯简化失败，降级直接检索: {e}")
        return _hybrid_recall(query, domain)

    return _hybrid_recall(query, domain)


def _rerank_context(question: str, candidates: list[dict]) -> tuple[str, list[dict]]:
    """精排层：Reranker 重排 + 相似度阈值过滤 + 组装 context 与 sources。"""
    cfg = get_config()

    # Reranker 重排序
    top_k = min(5, len(candidates))
    reranked = rerank(question, candidates, top_k=top_k)

    # 相似度阈值过滤：剔除低质量候选，避免无关内容进入 LLM 上下文。
    # 全部低于阈值时回退原始 Top-K（保证仍返回检索结果，避免空白回答）。
    threshold = cfg.retrieval.similarity_threshold
    filtered = [r for r in reranked if r.get("rerank_score", r.get("score", 0)) >= threshold]
    if not filtered:
        logger.warning(f"检索结果全部低于相似度阈值 {threshold}，回退原始 Top-{len(reranked)}")
    else:
        reranked = filtered

    def _q_label(r: dict) -> str:
        # 文档块的 question 是 chunk id，直接拼进 prompt 无意义且干扰 LLM
        return "文档片段" if is_chunk_id(r.get("question", "")) else r["question"]

    context = "\n\n".join(
        f"【问题】{_q_label(r)}（来源：{r.get('source', '')}）\n【回答】{r['answer']}"
        for r in reranked
    )
    # 按 token 预算截断 context，防止超大上下文溢出 LLM 窗口
    context = _truncate_context(context)
    sources = [
        {
            "question": r["question"],
            "score": round(r.get("rerank_score", r.get("score", 0)), 4),
            "source": r.get("source", ""),
            "source_url": r.get("source_url", ""),
        }
        for r in reranked
    ]
    return context, sources


def _market_source() -> dict:
    """行情上下文的溯源条目（type 便于前端/Agent 与知识库文档块区分展示）。

    数据源与数据时间都取自真实生效的那一次取数（provider 可切腾讯/东财），
    绝不写死——写死会让溯源信息与事实不符（曾出现"数据来自腾讯、标注写东财"）。
    """
    meta = last_snapshot_meta()
    stamp = f"（{meta['quote_time']}）" if meta.get("quote_time") else ""
    return {
        "question": f"实时行情快照{stamp}",
        "score": 1.0,
        "source": meta.get("source") or "公开行情接口",
        "source_url": meta.get("source_url", ""),
        "type": "realtime_market",
    }


def _build_context(question: str, intent, strategy: str | None = None) -> tuple[str, list[dict]]:
    """
    执行检索管线，返回 (context_str, sources_list)。

    策略路由（P0 多策略检索）：规则预筛（零开销）→ 可选 LLM 兜底 → 召回 → 精排。
    - strategy=None：按配置自动选择（multi_strategy_enabled 关闭时恒为直接检索）
    - 显式传入 strategy 时跳过选择器（评估/诊断脚本用）
    """
    domain = intent.domain if intent.domain != "general" else None
    cfg = get_config()

    # 实时行情注入（P5）：知识库是静态资料，行情类问句必然低相关（实测全部低于
    # 阈值回退 Top-5）。此处先取一份行情快照作为独立上下文块，块内自带来源与
    # 数据时间说明；任何失败静默降级（market_ctx 为空串，行为与升级前一致）。
    market_ctx = ""
    if cfg.market_data.enabled and is_realtime_query(question):
        market_ctx = build_market_context(question)
        if market_ctx:
            logger.info(f"实时行情已注入上下文（{len(market_ctx)} 字符）")
        else:
            logger.info("实时行情类问句未取到行情数据，回退纯知识库检索")

    if strategy is None:
        if cfg.retrieval.multi_strategy_enabled:
            strategy = select_strategy(question, llm_fallback=cfg.retrieval.strategy_llm_fallback)
        else:
            strategy = STRATEGY_DIRECT
    record_strategy(strategy)
    if strategy != STRATEGY_DIRECT:
        logger.info(f"检索策略命中: {strategy} (question={_mask_question(question)})")

    candidates = _recall_with_strategy(question, domain, strategy)
    if not candidates:
        # 行情类问句本就不依赖知识库命中：只有行情也要让 LLM 作答（否则会
        # 直接回"未在知识库中找到相关资料"，与用户预期不符）
        if market_ctx:
            logger.info("知识库无命中，仅注入实时行情上下文")
            record_retrieval("no_result")
            return market_ctx, [_market_source()]
        logger.warning("无检索结果，交由 LLM 直接回答")
        record_retrieval("no_result")
        return "", []

    context, sources = _rerank_context(question, candidates)
    if market_ctx:
        context = f"{market_ctx}\n\n{context}"
        sources = [_market_source()] + sources
    return context, sources


# ─── 老链路弃用标记 ────────────────────────────────────────────────────────────
# 融合 ChatBI/DocAudit 后，/chat/ 与 /chat/stream 只覆盖文档问答一条链路，
# 且带 BM25 早退（可能绕过上传文档）。保留是为了兼容老前端，新调用方一律走
# /chat/ask 与 /chat/ask_stream。这里只做提示，不改变行为，避免打断存量集成。
_DEPRECATION_NOTICE = (
    "此接口已弃用，请迁移到 /chat/ask（统一编排，含问数与文档审查能力）"
)
# HTTP 响应头仅允许 latin-1 字符，中文放头里会在 starlette 编码时直接崩
# （UnicodeEncodeError: 'latin-1' codec can't encode），头里只放 ASCII 提示。
_DEPRECATION_NOTICE_HEADER = (
    "Deprecated: migrate to /chat/ask (unified orchestration incl. NL2SQL and document review)"
)
_deprecation_logged = False


def _mark_deprecated(response: Response | None = None) -> None:
    """打弃用响应头；进程内只打一条 WARNING 日志，避免刷屏。"""
    global _deprecation_logged
    if response is not None:
        response.headers["Deprecation"] = "true"
        response.headers["X-Deprecated-Endpoint"] = "/chat/ -> /chat/ask"
        response.headers["Warning"] = f'299 - "{_DEPRECATION_NOTICE_HEADER}"'
    if not _deprecation_logged:
        _deprecation_logged = True
        logger.warning(
            "老链路 /chat/ 被调用：{}（后续不再告警）", _DEPRECATION_NOTICE
        )


# ─── 非流式接口 ────────────────────────────────────────────────────────────────
@router.post(
    "/",
    response_model=ChatResponse,
    summary="发送问题并获取回答（兼容接口，建议改用 /chat/ask）",
    deprecated=True,
)
def ask_question(
    req: ChatRequest,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    """
    非流式问答接口（老链路，**已降级为兼容接口**）。

    流程：Redis 缓存 → BM25 精确匹配 → 意图识别 → Milvus 向量检索 → Reranker → LLM

    与 /chat/ask 的差异（融合后不应再新增调用方）：
    - 不走 skill 路由，只做文档问答，**没有问数（NL2SQL）与文档审查能力**；
    - 有 BM25 早退，命中 FAQ 时可能完全绕过向量检索与上传文档；
    - 流式版本 /chat/stream 同理。

    新调用方请统一使用 /chat/ask（非流式）与 /chat/ask_stream（SSE）。
    """
    _mark_deprecated(response)
    return _legacy_answer(req, current_user)


def _legacy_answer(req: ChatRequest, current_user: dict) -> ChatResponse:
    """老链路实现：无 skill 路由（不含问数/审查），且带 BM25 早退。

    保留原因：① 兼容老前端；② [orchestrator] enable_answer_graph=false 时
    /chat/ask 回退到这里。新代码不应再调用——能力面比统一入口窄。
    """
    question = sanitize_input(req.message)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    logger.info(f"收到问题: {_mask_question(question)}")

    # 实时行情类问句：不进缓存、不走 BM25 早退。
    # 理由：行情随时变化，24h 答案缓存会把「某个时点的行情」冻住；而 BM25 命中的
    # 是静态 FAQ，答不了「现在多少点」，必须放行到检索+行情注入链路。
    realtime = is_realtime_query(question)

    # Step 1: Redis 缓存（行情类问句跳过读）
    cached = None if realtime else cache_get(question)
    if cached:
        logger.info("Redis 缓存命中")
        # 缓存命中仍保存到会话历史，保证多轮上下文完整（否则缓存轮次会从滑窗中丢失）
        if req.session_id:
            domain = cached.get("domain") or ""
            if req.save_user:
                _save_chat_message(req.session_id, "user", question, domain, user_id=current_user["user_id"])
            _save_chat_message(req.session_id, "assistant", cached.get("answer", ""), domain, user_id=current_user["user_id"])
        # 补回会话 ID（缓存不存 session_id，避免返回给前端时丢失）
        cached.setdefault("session_id", req.session_id)
        return ChatResponse(**cached)

    # Step 2: BM25 精确匹配
    # 早退阈值必须足够高：阈值过低时，语义沾边但答非所问的 FAQ 会直接返回，
    # 把 Milvus 里真正相关的文档块挡在检索之外。实测案例：
    #   「个人信用报告在哪里查询」→ 0.786 命中"减免征信报告查询费用"（答非所问）
    #   「存款保险的偿付限额」  → 0.912 命中"引入存款保险的建议理由"（答非所问）
    # 因此只有接近 1.0 的精确命中才早退，其余情况把 BM25 候选交给 Step 3 与
    # 文档块一起参与 Reranker 精排，由相关性决定最终答案。
    # 注意：softmax 达标只是"明显领先其他 FAQ"，还需 _bm25_early_gate_passed
    # 做一次绝对相关性校验，否则语义沾边但答非所问的 FAQ 会挡掉文档块。
    bm25 = get_bm25_retriever()
    bm25_best = bm25.get_best_match(question, threshold=0.75)
    if (not realtime and bm25_best
            and bm25_best["softmax_score"] >= BM25_EARLY_RETURN_THRESHOLD
            and _bm25_early_gate_passed(question, bm25_best)):
        answer = bm25_best["answer"]
        domain = bm25_best["category"]
        result = {
            "answer": answer,
            "domain": domain,
            "confidence": bm25_best["softmax_score"],
            "sources": [{"question": bm25_best["question"], "score": bm25_best["softmax_score"],
                         "source": bm25_best.get("source") or "bm25",
                         "source_url": bm25_best.get("source_url", "")}],
        }
        if not realtime:          # 行情类问句不写缓存（答案含时点数据）
            cache_set(question, result)
        logger.info("BM25 命中，直接返回答案")
        record_retrieval("bm25_early")
        # 保存问答到会话历史（与流式接口一致，支持多轮上下文理解）
        if req.session_id:
            if req.save_user:
                _save_chat_message(req.session_id, "user", question, domain, user_id=current_user["user_id"])
            _save_chat_message(req.session_id, "assistant", answer, domain, user_id=current_user["user_id"])
        return ChatResponse(**result)

    # Step 3: 意图识别
    intent = classify_intent(question)
    # 用户显式指定 domain 时优先采用（如前端已确定知识库场景），否则用意图识别结果
    domain = req.domain if req.domain else intent.domain
    confidence = intent.confidence
    logger.info(f"意图识别: domain={domain}, confidence={confidence:.4f}, hits={intent.keywords_hit}")

    # Step 3.5: 用户配额检查（fail-fast）。
    # 放在检索之前：超配额用户不再白耗 Milvus/BM25/Reranker 算力即可拿到 429。
    _check_llm_quota(current_user["user_id"])

    # 保存用户消息到会话历史（重新生成时跳过，避免重复）
    if req.session_id and req.save_user:
        _save_chat_message(req.session_id, "user", question, domain, user_id=current_user["user_id"])

    # Step 4-5: 检索管线（Milvus + BM25 融合 + Reranker）
    context, sources = _build_context(question, intent)
    record_retrieval("rag_pipeline")

    # 检索空结果兜底：context 为空说明知识库中无相关内容，
    # 直接返回友好提示而非让 LLM 凭空作答（防幻觉）。
    if not context:
        fallback_msg = "抱歉，未在知识库中找到与您问题相关的资料。请尝试换种问法或联系人工客服。"
        if req.session_id:
            _save_chat_message(req.session_id, "assistant", fallback_msg, domain, user_id=current_user["user_id"])
        return ChatResponse(
            answer=fallback_msg,
            domain=domain, confidence=0.0, sources=[],
            session_id=req.session_id,
        )

    # Step 6: LLM 生成回答（异常时降级返回检索结果，避免裸 500）
    # 配额已在 Step 3.5 前置检查
    # 加载对话历史，支持多轮上下文理解
    history = _load_recent_history(req.session_id, current_user["user_id"]) if req.session_id else ""
    try:
        answer = generate_answer(question, context, domain, history)
    except Exception as e:
        logger.error(f"LLM 生成回答失败，降级返回检索结果: {e}")
        # LLM 降级：提取检索 context 中首条 QA 的回答作为兜底答案，保证服务不中断
        fallback_answer = _extract_fallback_from_context(context)
        if req.session_id:
            _save_chat_message(req.session_id, "assistant", fallback_answer, domain, user_id=current_user["user_id"])
        return ChatResponse(
            answer=fallback_answer,
            domain=domain, confidence=0.0, sources=sources,
            session_id=req.session_id,
        )

    if req.session_id:
        _save_chat_message(req.session_id, "assistant", answer, domain, user_id=current_user["user_id"])

    result = {
        "answer": answer,
        "domain": domain,
        "confidence": round(confidence, 4),
        "sources": sources,
    }
    if not realtime:              # 行情类问句不写缓存（答案含时点数据）
        cache_set(question, result, ttl=86400)
    logger.info(f"回答生成完成，domain={domain}")

    return ChatResponse(
        answer=answer, domain=domain, confidence=confidence,
        sources=sources, session_id=req.session_id,
    )


# ─── 流式接口 ──────────────────────────────────────────────────────────────────
@router.post("/stream", summary="流式问答（SSE，兼容接口，建议改用 /chat/ask_stream）",
             deprecated=True)
def ask_question_stream(req: ChatRequest, response: Response,
                        current_user: dict = Depends(get_current_user)):
    """
    流式问答接口，实时推送 LLM 生成的 token。
    返回 Server-Sent Events (SSE) 格式。

    **已弃用**：不含 skill 路由（无问数/审查），建议迁移到 /chat/ask_stream。
    """
    _mark_deprecated(response)
    question = sanitize_input(req.message)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    user_id = current_user["user_id"]

    # 配额检查必须在开流之前：HTTPException 才能以标准 429 响应返回。
    # 若放进 event_generator，异常会被兜底逻辑吞掉变成"降级答案"，
    # 超配额用户永远感知不到配额限制。
    _check_llm_quota(user_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        yield 'data: {"event":"start","data":{}}\n\n'
        try:
            # 意图识别：BERT CPU 推理是阻塞调用，必须放线程池执行。
            # 直接在事件循环上跑会把所有并发请求（包括其他用户的 SSE token 流）串行化。
            intent = await asyncio.to_thread(classify_intent, question)
            # 用户显式指定 domain 时优先采用，否则用意图识别结果
            domain = req.domain if req.domain else intent.domain
            logger.info(f"流式问答 - 意图: domain={domain}")
            yield f"data: {json.dumps({'event': 'intent', 'data': {'domain': domain, 'confidence': intent.confidence}}, ensure_ascii=False)}\n\n"

            # 保存用户消息到会话历史（重新生成时跳过，避免重复）
            if req.session_id and req.save_user:
                await asyncio.to_thread(
                    _save_chat_message, req.session_id, "user", question, domain, user_id=user_id
                )

            # 检索管线（BGE-M3 编码 / Milvus 检索 / Reranker 均为阻塞 CPU 或同步 IO 调用）
            context, sources = await asyncio.to_thread(_build_context, question, intent)

            # 检索空结果兜底：直接返回提示，不调用 LLM（防幻觉 + 省 token）
            if not context:
                fallback_msg = "抱歉，未在知识库中找到与您问题相关的资料。请尝试换种问法或联系人工客服。"
                if req.session_id:
                    await asyncio.to_thread(
                        _save_chat_message, req.session_id, "assistant", fallback_msg, domain, user_id=user_id
                    )
                yield f"data: {json.dumps({'event': 'done', 'data': {'answer': fallback_msg}}, ensure_ascii=False)}\n\n"
                return

            if sources:
                yield f'data: {{"event":"sources","data":{json.dumps(sources, ensure_ascii=False)}}}\n\n'

            # 流式调用 LLM
            try:
                # Prompt 由 RAGPrompts.answer_messages 统一生成（system/user 分层，防注入），
                # 与非流式接口保持一致；history 支持多轮上下文理解（滑窗截断后传入）。
                history = (
                    await asyncio.to_thread(_load_recent_history, req.session_id, user_id)
                    if req.session_id else ""
                )
                messages = RAGPrompts.answer_messages(question, context, domain, history)
                cfg = get_config()
                # 使用 AsyncOpenAI 客户端，原生异步流式传输，避免 run_in_executor 逐 chunk 调度开销
                async_client = get_async_llm_client()
                record_llm(cfg.llm.model)
                response = await async_client.chat.completions.create(
                    model=cfg.llm.model,
                    messages=messages,
                    stream=True,
                    temperature=cfg.llm.temperature,
                    max_tokens=cfg.llm.max_tokens,
                )
                full_text = ""

                async for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full_text += delta
                        # 必须用 json.dumps 转义 delta 中的引号/换行，否则前端 JSON.parse 会失败
                        yield f"data: {json.dumps({'event': 'token', 'data': delta}, ensure_ascii=False)}\n\n"

                # 保存助手消息到会话历史，并写入 QA 缓存（与非流式接口行为一致，
                # 相同问题再问时非流式接口可直接命中缓存，避免重复 LLM 调用）
                if full_text:
                    if req.session_id:
                        await asyncio.to_thread(
                            _save_chat_message, req.session_id, "assistant", full_text, domain, user_id=user_id
                        )
                    await asyncio.to_thread(
                        cache_set, question,
                        {
                            "answer": full_text,
                            "domain": domain,
                            "confidence": round(intent.confidence, 4),
                            "sources": sources,
                        },
                        86400,
                    )

                yield f"data: {json.dumps({'event': 'done', 'data': {'answer': full_text}}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"LLM 流式调用失败，降级返回检索结果: {e}")
                # LLM 故障降级：提取检索 context 中首条 QA 的回答作为兜底答案，保证服务不中断。
                # 发 done 事件（前端正常渲染），带 degraded 标记供前端识别降级场景。
                fallback = _extract_fallback_from_context(context)
                if req.session_id:
                    await asyncio.to_thread(
                        _save_chat_message, req.session_id, "assistant", fallback, domain, user_id=user_id
                    )
                yield f"data: {json.dumps({'event': 'done', 'data': {'answer': fallback, 'degraded': True}}, ensure_ascii=False)}\n\n"

        except HTTPException as e:
            # 业务异常（如配额）：开流后无法再改状态码，转 error 事件交前端提示
            logger.warning(f"流式问答业务异常: {e.detail}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e.detail)}}, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 检索等阶段的未预期异常（如 Milvus/Redis 故障）：
            # 不兜底会让 SSE 流中途断掉，前端只能看到网络错误且无任何提示
            logger.exception(f"流式问答管线异常: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': '服务暂时不可用，请稍后重试'}}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── LangGraph 统一编排入口（融合 ChatBI/DocAudit, P3）────────────────────────
# orchestrator 在函数内局部 import，避免与 orchestrator.nodes -> routers.chat 形成循环依赖
@router.post("/search", summary="知识库检索（只检索不生成，MCP finrag.kb_search 后端）")
def search_knowledge_base(
    req: ChatRequest, top_k: int = 5, current_user: dict = Depends(get_current_user)
):
    """只做「意图分域 → 混合检索 → 精排」，返回带溯源的块列表，不调 LLM。

    供 MCP 工具层（finrag.kb_search）与外部 Agent 使用：Agent 先看检索结果
    决定是否够用/是否换问法，再决定是否要端到端回答（/chat/ask），
    这是 Agentic RAG「检索-评估-再检索」循环的基础设施。
    """
    question = sanitize_input(req.message)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    intent = classify_intent(question)
    context, sources = _build_context(question, intent, None)
    top_k = max(1, min(int(top_k or 5), 20))
    return ApiResponse(success=True, data={
        "hit_count": len(sources),
        "sources": sources[:top_k],
        "domain": intent.domain,
        "context_preview": (context or "")[:500],
    })


@router.post("/ask", summary="统一编排问答（LangGraph AnswerGraph 全 Skill 路由）")
async def ask_unified(req: ChatRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """LangGraph 统一入口：路由到 rag / nl2sql / chitchat，返回统一结构。

    配置 [orchestrator] enable_answer_graph=false 时回退到老链路 /chat/。
    """
    question = sanitize_input(req.message)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    cfg = get_config()
    if not cfg.orchestrator.enable_answer_graph:
        # enable_answer_graph=false 时回退老链路；此处是内部转发，不算"调用弃用接口"
        return _legacy_answer(req, current_user)

    _check_llm_quota(current_user["user_id"])

    from rag_qa.orchestrator.graph import run_ask
    # 纯规则预路由（零开销）：仅用于决定缓存域，避免 nl2sql 的旧答案污染 rag 结果
    from rag_qa.orchestrator.route import _decide_skill

    skill_hint = _decide_skill(question)
    # 行情类问句不进缓存：答案带时点数据，24h 缓存会返回过期行情
    realtime = is_realtime_query(question)
    cacheable = skill_hint in ("rag", "chitchat") and not realtime
    # key 带 role：内置/自定义知识库按角色鉴权，不加维度会让 admin 可见内容泄露给普通用户
    cache_key = f"ask:{skill_hint}:{current_user.get('role', 'user')}:{question}"

    if cacheable:
        cached = cache_get(cache_key)
        if cached:
            if req.session_id:
                _save_chat_message(
                    req.session_id, "assistant", cached.get("answer", ""),
                    cached.get("domain", ""), user_id=current_user["user_id"],
                )
            cached["trace"] = ["cache_hit"]
            return ApiResponse(success=True, data=cached)

    # 用户消息落会话（与老链路一致）：缺失会导致多轮指代消解拿不到上下文
    if req.session_id and req.save_user:
        _save_chat_message(
            req.session_id, "user", question, req.domain or "", user_id=current_user["user_id"]
        )

    final = await run_ask(
        question,
        role=current_user.get("role", "user"),
        session_id=req.session_id,
        user_id=current_user.get("user_id"),
        domain=req.domain,
        username=current_user.get("username", ""),
    )
    data = {
        "skill": final.get("skill", "rag"),
        "answer": final.get("answer", ""),
        "status": final.get("status", "ok"),
        "degraded": final.get("degraded", False),
        "sources": final.get("sources", []),
        "table": final.get("table"),
        "task": final.get("task"),
        "sql": final.get("sql", ""),
        "domain": final.get("domain", ""),
        "latency_ms": final.get("latency_ms", 0),
        "trace": final.get("trace", []),
    }
    # 仅 RAG/寒暄走缓存：问数结果依赖实时业务数据，审查结果带一次性 task_id。
    # 二次校验实际 skill——运行时路由可能用实体证据把预路由的 rag 精修成 nl2sql，
    # 只信 skill_hint 会把问数结果写进 rag 缓存域（24h 过期口径）。
    if cacheable and data["status"] == "ok" and data["skill"] in ("rag", "chitchat"):
        cache_set(cache_key, data, ttl=86400)
    # 助手消息落会话：否则下一轮追问（"再详细点"/"那它呢"）拿不到历史
    if req.session_id and data["answer"]:
        _save_chat_message(
            req.session_id, "assistant", data["answer"], data.get("domain", ""),
            user_id=current_user["user_id"],
        )
    try:
        from services.metrics import record_skill_route

        record_skill_route(data["skill"])
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        f"AnswerGraph: skill={data['skill']} status={data['status']} "
        f"latency={data['latency_ms']}ms question={_mask_question(question)}"
    )
    return ApiResponse(success=True, data=data)


@router.post("/ask_stream", summary="统一编排问答（SSE，节点级进度直通）")
def ask_unified_stream(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """LangGraph 流式统一入口：astream(updates) 逐节点进度 → SSE 事件。

    事件模型与老链路兼容：start / route / rag_retrieve / rag_answer / nl2sql /
    chitchat / done / error，data 为节点进度或最终 answer/table。
    """
    question = sanitize_input(req.message)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    _check_llm_quota(current_user["user_id"])
    user_id = current_user.get("user_id")
    if req.session_id and req.save_user:
        _save_chat_message(req.session_id, "user", question, req.domain or "", user_id=user_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        yield 'data: {"event":"start","data":{}}\n\n'
        final_answer = ""
        final_stage = ""
        try:
            from rag_qa.orchestrator.graph import stream_ask

            async for ev in stream_ask(
                question,
                role=current_user.get("role", "user"),
                session_id=req.session_id,
                user_id=user_id,
                domain=req.domain,
                username=current_user.get("username", ""),
            ):
                if ev["event"] == "done":
                    final_answer = (ev.get("data") or {}).get("answer", "")
                else:
                    final_stage = (ev.get("data") or {}).get("stage", final_stage)
                payload = json.dumps({"event": ev["event"], "data": ev["data"]}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except HTTPException as e:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e.detail)}}, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.exception(f"AnswerGraph 流式异常: {e}")
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': '服务暂时不可用，请稍后重试'}}, ensure_ascii=False)}\n\n"
        finally:
            # 与非流式 /ask 行为一致：会话历史必须落库，否则多轮追问丢失上下文
            if req.session_id and final_answer:
                _save_chat_message(
                    req.session_id, "assistant", final_answer,
                    req.domain or "", user_id=user_id,
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
