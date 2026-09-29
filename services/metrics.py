#!/usr/bin/env python
"""
services/metrics.py — Prometheus 指标定义与记录工具

集中定义 RAG 服务的关键指标，供 HTTP 中间件与业务代码埋点。
通过 /metrics 端点（main.py 注册）暴露给 Prometheus 抓取。

多进程部署（如 uvicorn --workers N）时，请将环境变量
PROMETHEUS_MULTIPROC_DIR 指向共享卷并在主进程调用 multiprocess 初始化，
否则各 worker 的计数各自独立。
"""

from prometheus_client import Counter, Histogram

# ─── HTTP 请求 ────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "finrag_http_requests_total",
    "HTTP 请求总数（按 method / path / status 维度）",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "finrag_http_request_duration_seconds",
    "HTTP 请求耗时（秒）",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

# ─── LLM 调用 ─────────────────────────────────────────────────────────────────
LLM_REQUEST_COUNT = Counter(
    "finrag_llm_requests_total",
    "LLM 调用次数（按模型维度）",
    ["model"],
)
LLM_TOKENS = Counter(
    "finrag_llm_tokens_total",
    "LLM token 用量（按模型 / 类型维度）",
    ["model", "type"],  # type ∈ {prompt, completion}
)

# ─── QA 缓存 ──────────────────────────────────────────────────────────────────
QA_CACHE_HIT = Counter("finrag_qa_cache_hits_total", "QA 回答缓存命中次数")
QA_CACHE_MISS = Counter("finrag_qa_cache_misses_total", "QA 回答缓存未命中次数")

# ─── 检索 ─────────────────────────────────────────────────────────────────────
RETRIEVAL_COUNT = Counter(
    "finrag_retrieval_requests_total",
    "检索管线执行次数（按通道维度）",
    ["channel"],  # channel ∈ {bm25_early, rag_pipeline, no_result}
)
RETRIEVAL_STRATEGY = Counter(
    "finrag_retrieval_strategy_total",
    "检索策略选择次数（按策略维度）",
    ["strategy"],  # strategy ∈ {直接检索, 假设问题检索, 子查询检索, 回溯问题检索}
)

# ─── 限流 ─────────────────────────────────────────────────────────────────────
RATE_LIMIT_HITS = Counter(
    "finrag_rate_limit_hits_total",
    "限流触发次数（按场景维度）",
    ["scope"],  # scope ∈ {auth, chat, llm_quota}
)

# ─── 融合能力（ChatBI 问数 / DocAudit 审查 / 统一编排）────────────────────────
SKILL_ROUTE = Counter(
    "finrag_skill_route_total",
    "统一编排路由的 Skill 分布",
    ["skill"],  # rag / nl2sql / chitchat / multimodal / review
)
NL2SQL_QUERY = Counter(
    "finrag_nl2sql_queries_total",
    "问数请求次数（按结果维度）",
    ["outcome"],  # data / refused / guard_rejected / failed / cached
)
NL2SQL_LATENCY = Histogram(
    "finrag_nl2sql_duration_seconds",
    "问数端到端耗时（秒）",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
REVIEW_TASK = Counter(
    "finrag_review_tasks_total",
    "文档审查任务计数（按阶段结果维度）",
    ["outcome"],  # accepted / failed / timeout / hitl_pending / completed
)


def record_request(method: str, path: str, status: int, latency_s: float) -> None:
    """HTTP 请求完成后记录计数与耗时。"""
    REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(latency_s)


def record_llm(model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """LLM 调用计数与 token 用量记录（流式响应无 usage，只记次数）。"""
    LLM_REQUEST_COUNT.labels(model=model).inc()
    if prompt_tokens:
        LLM_TOKENS.labels(model=model, type="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(model=model, type="completion").inc(completion_tokens)


def record_qa_cache(hit: bool) -> None:
    """QA 回答缓存命中 / 未命中计数。"""
    (QA_CACHE_HIT if hit else QA_CACHE_MISS).inc()


def record_retrieval(channel: str) -> None:
    """检索通道计数：bm25_early（早退直接命中）/ rag_pipeline（完整管线）/ no_result（无结果）。"""
    RETRIEVAL_COUNT.labels(channel=channel).inc()


def record_strategy(strategy: str) -> None:
    """检索策略选择计数（直接检索 / 假设问题检索 / 子查询检索 / 回溯问题检索）。"""
    RETRIEVAL_STRATEGY.labels(strategy=strategy).inc()


def record_rate_limit(scope: str) -> None:
    """限流触发计数。scope ∈ {auth, chat, llm_quota}。"""
    RATE_LIMIT_HITS.labels(scope=scope).inc()


def record_skill_route(skill: str) -> None:
    """统一编排的 Skill 路由分布（融合后判断各能力实际使用占比的依据）。"""
    SKILL_ROUTE.labels(skill=skill or "unknown").inc()


def record_nl2sql(outcome: str, latency_s: float = 0.0) -> None:
    """问数结果计数与耗时。outcome ∈ {data, refused, guard_rejected, failed, cached}。"""
    NL2SQL_QUERY.labels(outcome=outcome).inc()
    if latency_s > 0:
        NL2SQL_LATENCY.observe(latency_s)


def record_review_task(outcome: str) -> None:
    """审查任务计数。outcome ∈ {accepted, failed, timeout, hitl_pending, completed}。"""
    REVIEW_TASK.labels(outcome=outcome).inc()
