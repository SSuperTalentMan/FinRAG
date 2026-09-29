#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py — FastAPI 应用入口
组装所有路由、中间件、 lifespan 事件，启动服务。
"""

import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from config import get_config
from log_config import setup_logging
from routers import auth, chat, knowledge_base, document, conversation, admin, evaluation
from routers import review as review_router  # 多模态合规审查（融合 DocAudit, P1）
from routers import nl2sql as nl2sql_router   # 结构化问数（融合 ChatBI, P2）
from db.redis import get_redis, close_redis
from db.mysql import get_db, close_mysql_pool
from db.milvus import get_milvus_client, close_milvus, COLLECTION_NAME
from db.document import reconcile_stale_documents
from services import embedding, reranker
from rag_qa.core import query_classifier
from services.metrics import record_request, record_rate_limit
from services.rate_limiter import check_rate_limit
from services.reconciliation import start_reconciliation_loop

# ─── 初始化日志 ────────────────────────────────────────────────────────────────
setup_logging()

# 项目根目录（用于定位静态文件）
PROJECT_ROOT = Path(__file__).resolve().parent

# ─── 应用生命周期 ──────────────────────────────────────────────────────────────
async def lifespan(app: FastAPI):
    """应用启动 / 关闭时执行的操作。"""
    logger.info("=" * 50)
    logger.info("FinRag 服务启动中...")
    logger.info("=" * 50)

    cfg = get_config()
    logger.info(f"LLM 模型: {cfg.llm.model}")
    logger.info(f"BGE-M3 路径: {cfg.bge_m3_path}")
    logger.info(f"BGE-Reranker 路径: {cfg.bge_reranker_path}")
    logger.info(f"MySQL: {cfg.mysql.host}:{cfg.mysql.port}/{cfg.mysql.database}")
    logger.info(f"Redis: {cfg.redis.host}:{cfg.redis.port} db={cfg.redis.db}")
    logger.info(f"Milvus: {cfg.milvus.uri} db={cfg.milvus.database_name}")

    # 懒加载 BM25 索引（后台线程），避免阻塞启动
    import threading
    from services.bm25 import get_bm25_retriever
    def _load_bm25():
        try:
            get_bm25_retriever()
            logger.info("BM25 索引加载完成")
        except Exception as e:
            logger.warning(f"BM25 索引预加载失败（将在首次请求时重试）: {e}")
    t = threading.Thread(target=_load_bm25, daemon=True)
    t.start()
    logger.info("BM25 索引后台加载线程已启动")

    # 后台预加载检索模型（BGE-M3 / Reranker）与意图分类器：
    # 让首个请求不必承担模型加载延迟，同时让 /health/ready 能反映真实就绪状态。
    # 预加载失败仅告警，不阻塞启动（首次请求时会重试）。
    def _preload_model(label: str, loader) -> None:
        def _run():
            try:
                loader()
                logger.info(f"{label} 模型预加载完成")
            except Exception as e:
                logger.warning(f"{label} 模型预加载失败（将在首次请求时重试）: {e}")
        threading.Thread(target=_run, daemon=True, name=f"preload-{label}").start()

    _preload_model("BGE-M3", embedding.get_embedding_model)
    _preload_model("BGE-Reranker", reranker.get_reranker)
    _preload_model("BERT", query_classifier.get_classifier)
    logger.info("检索模型后台预加载线程已启动")

    # 确保审计日志表存在（幂等；兼容未执行 init.sql 的存量库）
    try:
        from db.mysql import ensure_audit_table
        ensure_audit_table()
        logger.info("审计日志表就绪")
    except Exception as e:
        logger.warning(f"审计日志表初始化失败（审计将跳过）: {e}")

    # 确保多模态合规审查表存在（融合 DocAudit, P1；幂等建表）
    try:
        from rag_qa.review import db as review_db
        review_db.ensure_tables()
        logger.info("合规审查表就绪")
    except Exception as e:
        logger.warning(f"合规审查表初始化失败（审查接口将不可用）: {e}")

    # 确保知识库属主列存在（幂等；属主权限模型依赖）
    try:
        from db.mysql import ensure_kb_owner_columns
        ensure_kb_owner_columns()
        logger.info("知识库属主列就绪")
    except Exception as e:
        logger.warning(f"知识库属主列迁移失败（属主校验将按缺失处理）: {e}")

    # 启动 MySQL↔Milvus 数据同步对账定时任务（间隔由配置控制，0=关闭）
    if cfg.reconciliation.sync_interval > 0:
        start_reconciliation_loop(cfg.reconciliation.sync_interval)

    # 清理孤儿文档状态：服务重启后 parsing 状态的文档永远不会完成，
    # 标记为 failed 让用户知道需要重新上传
    try:
        stale_count = reconcile_stale_documents()
        if stale_count > 0:
            logger.warning(f"检测到 {stale_count} 个中断的文档处理任务，已标记为 failed")
    except Exception as e:
        logger.warning(f"清理孤儿文档状态失败（不影响启动）: {e}")

    yield  # ← lifespan 在此处等待应用运行

    # ─── 优雅停机：释放所有外部连接 ──
    logger.info("FinRag 服务关闭，释放连接资源...")
    try:
        close_milvus()
    except Exception as e:
        logger.warning(f"关闭 Milvus 连接异常: {e}")
    try:
        close_redis()
    except Exception as e:
        logger.warning(f"关闭 Redis 连接异常: {e}")
    try:
        close_mysql_pool()
    except Exception as e:
        logger.warning(f"关闭 MySQL 连接池异常: {e}")
    # 问数（融合 ChatBI）用的是独立连接池（不同库 + 只读账号），需一并归还
    try:
        from rag_qa.nl2sql import db as nl2sql_db

        nl2sql_db._close_nl2sql_pools()
        logger.info("NL2SQL 连接池已释放")
    except Exception as e:
        logger.warning(f"关闭 NL2SQL 连接池异常: {e}")
    logger.info("FinRag 服务已停止")


# ─── 创建 FastAPI 应用 ─────────────────────────────────────────────────────────
app = FastAPI(
    title="FinRag 金融智能问答系统",
    description="基于 RAG 的金融领域智能问答服务，支持 MySQL BM25 + Milvus 向量检索 + BGE-Reranker 重排",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS 中间件 ───────────────────────────────────────────────────────────────
# 生产环境：显式来源，禁止 `*` 与 allow_credentials 同时出现（既不安全也会被浏览器拒绝）
cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.app.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# ─── 请求日志 / 请求ID / 限流中间件 ───────────────────────────────────────────
# Redis 分布式固定窗口限流（多实例共享状态，详见 services/rate_limiter.py）：
# - 认证接口：每 IP 每分钟限制（防爆破；验证码为主防线，限流兜底）
# - 问答接口：每 IP 每分钟限制（防单 IP 刷量）
# Redis 不可用时 fail-open，不阻塞业务。


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    # 存入 request.state，供全局异常处理器读取（异常发生在响应头设置之前）
    request.state.request_id = request_id
    client_ip = request.client.host if request.client else "unknown"
    start = time.perf_counter()
    path = request.url.path
    cfg = get_config()

    # 分布式限流（Redis；不可用时 fail-open）
    # 注意：路由统一挂载在 /api/v1 前缀下，用 endswith 兼容版本前缀演进
    scope = window = max_count = None
    if path.endswith("/auth/login") or path.endswith("/auth/register"):
        scope, window, max_count = "auth", cfg.rate_limit.auth_window, cfg.rate_limit.auth_max
    elif path.endswith("/chat/") or path.endswith("/chat/stream"):
        scope, window, max_count = "chat", cfg.rate_limit.chat_window, cfg.rate_limit.chat_max
    # 融合能力（P1 审查 / P2 问数）：统一入口 /chat/ask 走上面的 chat 限流，
    # 这里补独立端点——问数会打业务库，审查上传会跑 OCR/VL 流水线，都必须单独限量。
    elif path.endswith("/nl2sql/ask") or path.endswith("/nl2sql/execute"):
        scope, window, max_count = "nl2sql", cfg.rate_limit.chat_window, cfg.rate_limit.nl2sql_max
    elif path.endswith("/review/upload"):
        scope, window, max_count = "review_upload", cfg.rate_limit.chat_window, cfg.rate_limit.review_upload_max
    elif "/review/task/" in path or path.endswith("/review/rules"):
        scope, window, max_count = "review", cfg.rate_limit.chat_window, cfg.rate_limit.review_max

    if scope:
        allowed, count = check_rate_limit(f"ip:{client_ip}:{scope}", max_count, window)
        if not allowed:
            logger.warning(f"限流触发: scope={scope} ip={client_ip} count={count} path={path}")
            record_rate_limit(scope)
            return JSONResponse(
                status_code=429,
                content={"success": False, "message": "请求过于频繁，请稍后再试"},
                headers={"X-Request-ID": request_id},
            )

    try:
        response = await call_next(request)
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        logger.error(f"[{request_id}] 未处理异常 {path}: {e} ({elapsed:.1f}ms)")
        raise

    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    # Prometheus 指标：跳过 /metrics 自身避免循环抓取干扰。
    # 标签用路由模板（如 /document/{doc_id}）而非原始路径：
    # 原始路径里的每个动态 ID 都会产生新的 label 组合，指标基数随业务数据无限增长，
    # 最终拖垮 Prometheus 内存与查询性能。
    # 注意：新版 FastAPI include_router 为懒包含，scope["route"] 是内部 APIRoute，
    # 其 path 不含 /api/v1 前缀——前缀是全应用常量、不参与基数控制，无需强行还原。
    if path != "/metrics":
        route = request.scope.get("route")
        metric_path = route.path if route is not None else path
        record_request(request.method, metric_path, response.status_code, elapsed / 1000.0)
    logger.info(
        f"[{request_id}] {request.method} {path} -> {response.status_code} ({elapsed:.1f}ms) ip={client_ip}"
    )
    return response


# ─── 全局异常处理器（结构化错误响应，带 request_id 关联日志）────────────────────
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数校验失败（422）：返回结构化错误，带 request_id。"""
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID", "unknown")
    logger.warning(f"[{request_id}] 参数校验失败 {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "请求参数校验失败",
            "errors": jsonable_encoder(exc.errors()),
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """未捕获异常兜底（500）：记录完整堆栈，返回结构化错误，避免裸 500。"""
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID", "unknown")
    logger.error(
        f"[{request_id}] 未处理异常 {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误，请稍后重试",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


# ─── 注册路由 ──────────────────────────────────────────────────────────────────
# 所有 API 路由统一挂载在 /api/v1 前缀下，便于未来版本演进与灰度发布
_API_PREFIX = "/api/v1"
app.include_router(auth.router, prefix=_API_PREFIX)
app.include_router(chat.router, prefix=_API_PREFIX)
app.include_router(knowledge_base.router, prefix=_API_PREFIX)
app.include_router(document.router, prefix=_API_PREFIX)
app.include_router(conversation.router, prefix=_API_PREFIX)
app.include_router(admin.router, prefix=_API_PREFIX)
app.include_router(evaluation.router, prefix=_API_PREFIX)
app.include_router(review_router.router, prefix=_API_PREFIX)  # 多模态合规审查（融合 DocAudit, P1）
app.include_router(nl2sql_router.router, prefix=_API_PREFIX)   # 结构化问数（融合 ChatBI, P2）

# ─── 静态文件（前端页面）────────────────────────────────────────────────────────
_static_dir = PROJECT_ROOT / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
def root():
    """根路径返回前端单页应用。"""
    return FileResponse(str(_static_dir / "index.html"))


# ─── 健康检查 ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["健康检查"])
def health():
    """浅健康检查（探针用，不探外部依赖）。"""
    cfg = get_config()
    return {
        "status": "healthy",
        "service": "FinRag",
        "version": "1.0.0",
        "llm_model": cfg.llm.model,
        "embedding_dim": cfg.bge_m3_dim,
    }


@app.get("/health/ready", tags=["健康检查"])
def health_ready():
    """深度就绪检查：探测 DB 连通 + 检索/重排模型就绪，供 k8s readinessProbe 使用。

    - MySQL / Redis / Milvus：连通性探测
    - BGE-M3 / Reranker：检索核心，未就绪则服务未就绪（避免模型未加载时导流）
    - BERT：意图分类，缺失时可降级为关键词分类，故不阻断就绪
    """
    checks: dict[str, str] = {}

    # Redis
    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"down: {e}"

    # MySQL
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        checks["mysql"] = "ok"
    except Exception as e:
        checks["mysql"] = f"down: {e}"

    # Milvus
    try:
        client = get_milvus_client()
        client.has_collection(COLLECTION_NAME)
        checks["milvus"] = "ok"
    except Exception as e:
        checks["milvus"] = f"down: {e}"

    # 模型就绪状态（只读探测，不触发加载）
    try:
        checks["bge_m3"] = "ok" if embedding.is_model_loaded() else "loading"
        checks["reranker"] = "ok" if reranker.is_model_loaded() else "loading"
        checks["bert"] = "ok" if query_classifier.is_classifier_ready() else "degraded(关键词兜底)"
    except Exception as e:
        checks["bge_m3"] = checks["reranker"] = checks["bert"] = f"error: {e}"

    # 关键依赖（DB + 检索/重排模型）全部 ok 才算就绪；BERT 可降级不阻断
    critical = ("redis", "mysql", "milvus", "bge_m3", "reranker")
    all_ok = all(checks.get(k) == "ok" for k in critical)
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )


# ─── Prometheus 指标 ──────────────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False, tags=["可观测性"])
def metrics():
    """暴露 Prometheus 指标，供抓取（如 Prometheus / Grafana）。

    多进程部署（uvicorn --workers N / gunicorn）：设置环境变量
    PROMETHEUS_MULTIPROC_DIR 指向共享目录后，自动切换为多进程聚合采集，
    各 worker 的指标在抓取时合并（工作进程退出时需配合 marker 清理，见
    prometheus_client 文档 multiprocess 模式）。
    """
    import os
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import CollectorRegistry, multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─── 启动入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run(
        "main:app",
        host=cfg.app.host,
        port=cfg.app.port,
        reload=False,
        log_level="info",
    )
