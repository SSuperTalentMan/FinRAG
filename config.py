#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py — 配置管理模块
从 config.ini 读取所有配置，并提供类型安全的访问接口。

优先级（高 → 低）：
  1. 环境变量（便于容器化 / 不同部署环境覆盖，不与代码耦合）
  2. config.ini
  3. 内置默认值

这样在开发机（宿主机已运行 Redis/MySQL/Milvus）与 Docker / K8s 中都能无缝切换。
"""

import os
import json
import threading
import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# 项目根目录：始终基于本文件位置推导，避免把绝对路径（如 D:\FinRag）写死
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "config.ini"


def _env(name: str, default: str) -> str:
    """优先读取环境变量，缺失时回退到 default。"""
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _get(cp, section: str, key: str, default="") -> str:
    """从 ConfigParser 安全取值（返回字符串）。"""
    return cp.get(section, key, fallback=default)


def _int(cp, section: str, key: str, default: int = 0) -> int:
    return int(cp.get(section, key, fallback=str(default)))


def _float(cp, section: str, key: str, default: float = 0.0) -> float:
    return float(cp.get(section, key, fallback=str(default)))


def _bool(cp, section: str, key: str, default: bool = False) -> bool:
    return cp.get(section, key, fallback=str(default)).lower() in ("true", "1", "yes")


def _env_bool(name: str, default: bool) -> bool:
    """环境变量优先的布尔读取：未设置时返回 default。"""
    val = os.getenv(name)
    if val in (None, ""):
        return default
    return val.lower() in ("1", "true", "yes")


def _json_list(cp, section: str, key: str, default=None) -> list:
    raw = cp.get(section, key, fallback=None)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default or []
    return default or []


def _resolve_path(path_str: str) -> str:
    """相对路径基于项目根目录解析，绝对路径原样返回。"""
    p = Path(path_str)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


# ─── 配置数据类 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = "root"
    database: str = "finrag_qa"
    charset: str = "utf8mb4"


@dataclass(frozen=True)
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0


@dataclass(frozen=True)
class MilvusConfig:
    uri: str = "http://localhost:19530"
    token: str = ""        # 本地 standalone 默认无鉴权；为空时不传 token
    database_name: str = "finrag"
    collection_name: str = "finrag_faq"


@dataclass(frozen=True)
class LLMConfig:
    model: str = "qwen3.7-flash"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.7
    max_tokens: int = 2048
    # 备用模型：主模型调用失败（限流/下线/网络）时降级重试一次；空串=不启用
    fallback_model: str = ""
    # 多模态融合新增：结构化任务模型（抽取/比对/审查）/ 轻量模型（分级）/ 视觉兜底模型
    sql_model: str = ""
    fast_model: str = ""
    vl_model: str = "qwen-vl-max"
    timeout_seconds: int = 180
    max_retries: int = 2


@dataclass(frozen=True)
class MultimodalConfig:
    """多模态分级解析参数（融合 DocAudit parsing）。"""
    digital_min_chars: int = 50       # 页文本层字符数阈值，低于走 OCR
    ocr_confidence_route: float = 0.85  # OCR 平均置信度低于此 → qwen-vl 兜底
    page_concurrency: int = 2         # 页级并行度（16G 单机保守）
    upload_max_mb: int = 50
    uploads_dir: str = "./data/compliance/uploads"


@dataclass(frozen=True)
class ComplianceConfig:
    """条条款抽取 + LangGraph 合规审查参数（融合 DocAudit review）。"""
    coverage_threshold: float = 0.80  # 条款内容页码回验通过率
    extract_max_retries: int = 2
    retrieve_top_k: int = 3           # 规则召回调优后条数
    batch_size: int = 4               # 每批审查条款数（LangGraph 循环粒度）
    dual_temp: bool = True            # violation 用温度 0.7 复核一次，冲突转人工
    reports_dir: str = "./data/compliance/reports"
    pipeline_timeout_seconds: int = 1800  # 单任务流水线超时（0=不限）；防 LLM 挂起导致任务永久卡中间态


@dataclass(frozen=True)
class NL2SqlConfig:
    """结构化数据问数（融合 ChatBI, P2）：业务库只读执行 + SQLGuard 纵深防御。

    mysql 通用连接参数沿用 [mysql]（宿主/端口/管理员），这里只定义业务库与只读账号。
    只读账号（chatbi_ro）仅授 SELECT 权限，加上会话级 READ ONLY 双保险。
    """
    biz_database: str = "biz_demo"           # 业务库（只读账号可 SELECT）
    meta_database: str = "chatbi_meta"       # 元数据库（表/字段/指标口径）
    ro_user: str = "chatbi_ro"               # 业务库只读账号
    # 只读账号密码：刻意不设默认值——缺省即报"未配置"，避免用弱口令悄悄连库
    ro_password: str = ""
    max_rows: int = 1000                     # SQLGuard 强制 LIMIT 上限
    timeout_ms: int = 15000                  # 查询超时（会话级 + SELECT hint 双兜底；重聚合如退货率需放宽）
    explain_scan_threshold: int = 2_000_000  # EXPLAIN 预检扫描行数阈值，超过拒绝
    repair_max_rounds: int = 3               # SQL 生成失败自修复重写轮数（JOIN/截断定向修复用）
    # 单次 SQL 生成的硬超时：防止 LLM 挂起把端到端延迟堆到 38s+（评估中 3 道 in-schema error 根因）
    gen_timeout_seconds: int = 30
    # 单条 SQL 生成最大 token：防止模型输出过长被截断，导致 JSON 解析失败（Unterminated string）
    gen_max_tokens: int = 1500
    schema_top_k: int = 4                    # Schema（表/字段）召回条数
    embedding_model: str = "qwen3.7-text-embedding-flash"  # 阿里云百炼 API Embedding
    embedding_dim: int = 1024
    use_embedding_recall: bool = True        # Schema 召回是否启用向量召回（不启用则纯关键词）
    # 问数结果缓存（秒）。业务数据有时效，不能沿用 QA 的 24h；0=关闭。
    # 缓存键走 finrag:qa: 前缀，文档变更时的 clear_qa_cache 会一并清掉。
    cache_ttl_seconds: int = 600


@dataclass(frozen=True)
class OrchestratorConfig:
    """统一编排层（融合 ChatBI/DocAudit, P3）：LangGraph 路由到 rag / nl2sql / chitchat。
    老链路（routers/chat.py 直接管线）仍保留，True 时 /chat/ask 走 AnswerGraph。"""
    enable_answer_graph: bool = True   # /chat/ask 走 LangGraph AnswerGraph；False=回退老链路
    route_llm_fallback: bool = False   # 规则未命中时是否用 LLM 精判（多一次调用，默认关）
    answer_top_k: int = 5              # RAG Skill 精排返回 Top-K
    route_model: str = ""              # 路由/兜底用模型（空=主模型）
    # 数据驱动路由：问题命中的业务实体加权分低于此值 → 判 rag（消除泛词误判成问数）
    route_entity_min_score: float = 2.0
    # 对话内直接审查：粘贴文本超过此长度仍引导上传（避免一次审查烧掉大量 token）
    review_inline_max_chars: int = 8000
    review_enabled: bool = True        # 是否把「文档审查」纳入统一编排路由
    # Agentic RAG（P4）：rag/nl2sql 分支改走 Agent 循环（LLM 自主决定检索/问数/收尾，
    # 支持多跳检索与跨工具组合）；失败自动回退固定路由。默认关——金融场景确定性优先。
    agentic_mode: bool = False
    agentic_max_hops: int = 3          # Agent 循环最大工具调用次数（含收尾前）


@dataclass(frozen=True)
class MarketDataConfig:
    """实时行情注入（P5）：静态知识库无法回答"现在的行情如何"。

    仅对识别为「实时行情类」的问句注入行情快照，任何失败均静默降级。
    """
    enabled: bool = True
    # 数据源：tencent（主，抗 urllib 直连）| eastmoney（备）| auto（依次尝试）
    provider: str = "auto"
    tencent_url: str = "https://qt.gtimg.cn/q="
    # 腾讯代码规则：sh=上交所 sz=深交所；顺序即展示顺序
    tencent_symbols: str = "sh000001,sz399001,sz399006,sh000300,sh000688"
    eastmoney_url: str = "https://push2.eastmoney.com/api/qt/ulist.np/get"
    # 东财 secid 规则：1.=上交所 0.=深交所
    eastmoney_symbols: str = "1.000001,0.399001,0.399006,1.000300,1.000688"
    timeout_seconds: float = 3.0
    cache_ttl_seconds: int = 30


@dataclass(frozen=True)
class RetrievalConfig:
    parent_chunk_size: int = 1200
    child_chunk_size: int = 300
    chunk_overlap: int = 50
    retrieval_k: int = 5
    candidate_m: int = 5
    similarity_threshold: float = 0.7
    bm25_hit_threshold: float = 0.75
    # 多策略检索（直接/HyDE/子查询/回溯）：规则预筛路由，绝大多数查询仍走直接检索；
    # 关闭后全部走直接检索（旧行为）
    multi_strategy_enabled: bool = True
    # 规则未命中时对长查询启用 LLM 策略选择器兜底（增加一次 LLM 调用延迟，默认关闭）
    strategy_llm_fallback: bool = False


@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    valid_sources: list[str] = field(default_factory=lambda: [
        "banking", "corporate_finance", "financial_accounting",
        "financial_markets", "fintech", "insurance",
        "investment_banking", "personal_finance", "risk_management", "stock_market",
    ])
    customer_service_phone: str = "400-888-8888"
    documents_path: str = "./data/documents"
    faq_data_path: str = "./data/faq"
    # 允许跨域的前端来源（生产环境务必显式配置，不要使用 * 配合凭据）
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000",
    ])


@dataclass(frozen=True)
class ModelConfig:
    bert_intent_model_path: str = "./rag_qa/models/bert_intent"
    bert_base_model_path: str = "./bert-base-chinese"
    bert_epochs: int = 5
    bert_batch_size: int = 16
    bert_lr: float = 2e-5
    bert_max_length: int = 64
    bert_min_general_samples: int = 50


@dataclass(frozen=True)
class RateLimitConfig:
    """限流 / 配额参数（Redis 分布式固定窗口）。"""
    auth_window: int = 60         # 认证接口窗口（秒）
    auth_max: int = 20            # 认证接口窗口内最大次数（每 IP）
    chat_window: int = 60         # 问答接口全局窗口（秒）
    chat_max: int = 30            # 问答接口窗口内最大次数（每 IP）
    llm_quota_window: int = 3600  # LLM 用户配额窗口（秒）
    llm_quota_max: int = 100      # 用户配额窗口内最大问答次数
    # 融合能力（ChatBI 问数 / DocAudit 审查）独立限流：这两类比普通问答更重
    # （问数打业务库 + LLM，审查跑 OCR/VL 流水线），不能与普通问答共用配额
    nl2sql_max: int = 30          # 问数接口窗口内最大次数（每 IP）
    review_max: int = 60          # 审查查询接口窗口内最大次数（每 IP）
    review_upload_max: int = 10   # 审查上传接口窗口内最大次数（每 IP，重任务）


@dataclass(frozen=True)
class ReconciliationConfig:
    """MySQL↔Milvus 数据一致性对账参数。"""
    sync_interval: int = 3600    # 对账间隔（秒），0=关闭定时对账
    auto_clean: bool = False     # 是否自动清理孤儿向量（默认 false，仅告警人工介入）
    alert_threshold: int = 10    # 差异数量告警阈值（超出则升级为告警级日志）


class Config:
    """统一配置入口，所有服务从单例获取配置。"""

    def __init__(self, config_path: Path = CONFIG_PATH):
        cp = configparser.ConfigParser()
        cp.read(config_path, encoding="utf-8")

        # ---- MySQL（环境变量优先）----
        self.mysql = MySQLConfig(
            host     = _env("MYSQL_HOST",     _get(cp, "mysql", "host", "localhost")),
            port     = int(_env("MYSQL_PORT", str(_int(cp, "mysql", "port", 3306)))),
            user     = _env("MYSQL_USER",     _get(cp, "mysql", "user", "root")),
            password = _env("MYSQL_PASSWORD", _get(cp, "mysql", "password", "root")),
            database = _env("MYSQL_DATABASE", _get(cp, "mysql", "database", "finrag_qa")),
            charset  = _get(cp, "mysql", "charset", "utf8mb4"),
        )

        # ---- Redis（环境变量优先）----
        self.redis = RedisConfig(
            host     = _env("REDIS_HOST",     _get(cp, "redis", "host", "localhost")),
            port     = int(_env("REDIS_PORT", str(_int(cp, "redis", "port", 6379)))),
            password = _env("REDIS_PASSWORD", _get(cp, "redis", "password", "")),
            db       = int(_env("REDIS_DB",   str(_int(cp, "redis", "db", 0)))),
        )

        # ---- Milvus（环境变量优先）----
        mv_uri  = _env("MILVUS_URI", f"http://{_get(cp, 'milvus', 'host', 'localhost')}:{_int(cp, 'milvus', 'port', 19530)}")
        mv_user = _env("MILVUS_USER", _get(cp, "milvus", "user", "root"))
        mv_pwd  = _env("MILVUS_PASSWORD", _get(cp, "milvus", "password", ""))
        mv_token = _env("MILVUS_TOKEN", f"{mv_user}:{mv_pwd}" if mv_pwd else "")
        self.milvus = MilvusConfig(
            uri             = mv_uri,
            token           = mv_token,
            database_name   = _env("MILVUS_DATABASE",   _get(cp, "milvus", "database_name", "finrag")),
            collection_name = _env("MILVUS_COLLECTION", _get(cp, "milvus", "collection_name", "finrag_faq")),
        )

        # ---- LLM（API Key 仅来自环境变量 / .env，绝不写死在配置文件里）----
        llm_api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("API_KEY")
            or _get(cp, "llm", "dashscope_api_key", _get(cp, "llm", "api_key", ""))
        )
        llm_base_url = _env(
            "DASHSCOPE_BASE_URL",
            _get(cp, "llm", "dashscope_base_url",
                 _get(cp, "llm", "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
        )
        self.llm = LLMConfig(
            model       = _env("LLM_MODEL", _get(cp, "llm", "model", "qwen3.7-flash")),
            api_key     = llm_api_key,
            base_url    = llm_base_url,
            temperature = _float(cp, "llm", "temperature", 0.7),
            max_tokens  = _int(cp, "llm", "max_tokens", 2048),
            fallback_model = _env("LLM_FALLBACK_MODEL", _get(cp, "llm", "fallback_model", "")),
            # 多模态融合：结构化任务模型
            sql_model  = _env("LLM_SQL_MODEL", _get(cp, "llm", "sql_model", "")),
            fast_model = _env("LLM_FAST_MODEL", _get(cp, "llm", "fast_model", "")),
            vl_model   = _env("LLM_VL_MODEL", _get(cp, "llm", "vl_model", "qwen-vl-max")),
            timeout_seconds = _int(cp, "llm", "timeout_seconds", 180),
            max_retries     = _int(cp, "llm", "max_retries", 2),
        )

        # ---- 多模态解析 / 合规审查（融合 DocAudit P1）----
        self.multimodal = MultimodalConfig(
            digital_min_chars    = _int(cp, "multimodal", "digital_min_chars", 50),
            ocr_confidence_route = _float(cp, "multimodal", "ocr_confidence_route", 0.85),
            page_concurrency     = _int(cp, "multimodal", "page_concurrency", 2),
            upload_max_mb        = _int(cp, "multimodal", "upload_max_mb", 50),
            uploads_dir          = _resolve_path(_get(cp, "multimodal", "uploads_dir",
                                                     "./data/compliance/uploads")),
        )
        self.compliance = ComplianceConfig(
            coverage_threshold = _float(cp, "compliance", "coverage_threshold", 0.80),
            extract_max_retries = _int(cp, "compliance", "extract_max_retries", 2),
            retrieve_top_k      = _int(cp, "compliance", "retrieve_top_k", 3),
            batch_size          = _int(cp, "compliance", "batch_size", 4),
            dual_temp           = _bool(cp, "compliance", "dual_temp", True),
            reports_dir         = _resolve_path(_get(cp, "compliance", "reports_dir",
                                                     "./data/compliance/reports")),
        )

        # ---- 结构化问数（融合 ChatBI, P2）----
        self.nl2sql = NL2SqlConfig(
            biz_database     = _get(cp, "nl2sql", "biz_database", "biz_demo"),
            meta_database    = _get(cp, "nl2sql", "meta_database", "chatbi_meta"),
            ro_user          = _get(cp, "nl2sql", "ro_user", "chatbi_ro"),
            # 不回落到硬编码口令：未配置时由 rag_qa/nl2sql/db.py 在首次连库时 fail-fast
            ro_password      = _env("NL2SQL_RO_PASSWORD", _get(cp, "nl2sql", "ro_password", "")),
            max_rows         = _int(cp, "nl2sql", "max_rows", 1000),
            timeout_ms       = _int(cp, "nl2sql", "timeout_ms", 5000),
            explain_scan_threshold = _int(cp, "nl2sql", "explain_scan_threshold", 2000000),
            repair_max_rounds = _int(cp, "nl2sql", "repair_max_rounds", 2),
            schema_top_k     = _int(cp, "nl2sql", "schema_top_k", 4),
            embedding_model  = _env("NL2SQL_EMBEDDING_MODEL",
                                   _get(cp, "nl2sql", "embedding_model", "qwen3.7-text-embedding-flash")),
            embedding_dim    = _int(cp, "nl2sql", "embedding_dim", 1024),
            use_embedding_recall = _bool(cp, "nl2sql", "use_embedding_recall", True),
            cache_ttl_seconds = _int(cp, "nl2sql", "cache_ttl_seconds", 600),
        )

        # ---- 统一编排（融合 ChatBI/DocAudit, P3）----
        self.orchestrator = OrchestratorConfig(
            enable_answer_graph = _env_bool(
                "ORCH_ENABLE_ANSWER_GRAPH", _bool(cp, "orchestrator", "enable_answer_graph", True)
            ),
            route_llm_fallback  = _env_bool(
                "ORCH_ROUTE_LLM_FALLBACK", _bool(cp, "orchestrator", "route_llm_fallback", False)
            ),
            answer_top_k        = _int(cp, "orchestrator", "answer_top_k", 5),
            route_model         = _get(cp, "orchestrator", "route_model", ""),
            route_entity_min_score = float(
                _env("ORCH_ROUTE_ENTITY_MIN_SCORE",
                     _get(cp, "orchestrator", "route_entity_min_score", "2.0"))),
            review_inline_max_chars = _int(cp, "orchestrator", "review_inline_max_chars", 8000),
            review_enabled      = _env_bool(
                "ORCH_REVIEW_ENABLED", _bool(cp, "orchestrator", "review_enabled", True)
            ),
            agentic_mode        = _env_bool(
                "ORCH_AGENTIC_MODE", _bool(cp, "orchestrator", "agentic_mode", False)
            ),
            agentic_max_hops    = _int(cp, "orchestrator", "agentic_max_hops", 3),
        )

        # ---- 实时行情（P5：行情类问句注入实时数据）----
        self.market_data = MarketDataConfig(
            enabled = _env_bool(
                "MARKET_DATA_ENABLED", _bool(cp, "market_data", "enabled", True)
            ),
            provider = _env("MARKET_DATA_PROVIDER",
                            _get(cp, "market_data", "provider", "auto")),
            tencent_url = _env("MARKET_DATA_TENCENT_URL",
                               _get(cp, "market_data", "tencent_url", "https://qt.gtimg.cn/q=")),
            tencent_symbols = _env("MARKET_DATA_TENCENT_SYMBOLS",
                                   _get(cp, "market_data", "tencent_symbols",
                                        "sh000001,sz399001,sz399006,sh000300,sh000688")),
            eastmoney_url = _env("MARKET_DATA_EASTMONEY_URL",
                                 _get(cp, "market_data", "eastmoney_url",
                                      "https://push2.eastmoney.com/api/qt/ulist.np/get")),
            eastmoney_symbols = _env("MARKET_DATA_EASTMONEY_SYMBOLS",
                                     _get(cp, "market_data", "eastmoney_symbols",
                                          "1.000001,0.399001,0.399006,1.000300,1.000688")),
            timeout_seconds = float(_env("MARKET_DATA_TIMEOUT",
                                         _get(cp, "market_data", "timeout_seconds", "3.0"))),
            cache_ttl_seconds = _int(cp, "market_data", "cache_ttl_seconds", 30),
        )

        # ---- Retrieval ----
        self.retrieval = RetrievalConfig(
            parent_chunk_size    = _int(cp, "retrieval", "parent_chunk_size", 1200),
            child_chunk_size     = _int(cp, "retrieval", "child_chunk_size", 300),
            chunk_overlap        = _int(cp, "retrieval", "chunk_overlap", 50),
            retrieval_k          = _int(cp, "retrieval", "retrieval_k", 5),
            candidate_m          = _int(cp, "retrieval", "candidate_m", 5),
            similarity_threshold = _float(cp, "retrieval", "similarity_threshold", 0.7),
            bm25_hit_threshold   = _float(cp, "retrieval", "bm25_hit_threshold", 0.75),
            multi_strategy_enabled = _env_bool(
                "RETRIEVAL_MULTI_STRATEGY", _bool(cp, "retrieval", "multi_strategy_enabled", True)
            ),
            strategy_llm_fallback = _env_bool(
                "RETRIEVAL_STRATEGY_LLM_FALLBACK", _bool(cp, "retrieval", "strategy_llm_fallback", False)
            ),
        )

        # ---- App ----
        cors_raw = _env("CORS_ORIGINS", _get(cp, "app", "cors_origins", ""))
        cors_list = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else [
            "http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000",
        ]
        self.app = AppConfig(
            host                    = _env("APP_HOST", _get(cp, "app", "host", "0.0.0.0")),
            port                    = int(_env("APP_PORT", str(_int(cp, "app", "port", 8000)))),
            valid_sources           = _json_list(cp, "app", "valid_sources") or [
                "banking", "corporate_finance", "financial_accounting",
                "financial_markets", "fintech", "insurance",
                "investment_banking", "personal_finance", "risk_management", "stock_market",
            ],
            customer_service_phone  = _get(cp, "app", "customer_service_phone", "400-888-8888"),
            documents_path          = _resolve_path(_get(cp, "app", "documents_path", "./data/documents")),
            faq_data_path           = _resolve_path(_get(cp, "app", "faq_data_path", "./data/faq")),
            cors_origins            = cors_list,
        )

        # ---- 本地模型路径（始终基于项目根目录，容器/宿主机通用）----
        self.bge_m3_path:        str = str(PROJECT_ROOT / "bge-m3")
        self.bge_reranker_path:  str = str(PROJECT_ROOT / "bge-reranker-v2-m3")
        self.bge_m3_dim:         int = 1024

        # ---- 日志 ----
        self.log_file: str  = _resolve_path(_get(cp, "logger", "log_file", str(PROJECT_ROOT / "logs" / "app.log")))
        self.log_level: str = _env("LOG_LEVEL", _get(cp, "logger", "log_level", "INFO"))

        # ---- BERT 意图分类模型配置 ----
        self.models = ModelConfig(
            bert_intent_model_path  = _resolve_path(_get(cp, "models", "bert_intent_model_path",
                                          "./rag_qa/models/bert_intent")),
            bert_base_model_path    = _resolve_path(_get(cp, "models", "bert_base_model_path",
                                          "./bert-base-chinese")),
            bert_epochs             = _int(cp, "models", "bert_epochs", 5),
            bert_batch_size         = _int(cp, "models", "bert_batch_size", 16),
            bert_lr                 = _float(cp, "models", "bert_lr", 2e-5),
            bert_max_length         = _int(cp, "models", "bert_max_length", 64),
            bert_min_general_samples = _int(cp, "models", "bert_min_general_samples", 50),
        )

        # 兼容旧字段名（部分脚本可能直接读取）
        self.api_key: str = self.llm.api_key

        # ---- 限流 / 配额 ----
        self.rate_limit = RateLimitConfig(
            auth_window      = int(_env("RATE_LIMIT_AUTH_WINDOW",      str(_int(cp, "rate_limit", "auth_window", 60)))),
            auth_max         = int(_env("RATE_LIMIT_AUTH_MAX",         str(_int(cp, "rate_limit", "auth_max", 20)))),
            chat_window      = int(_env("RATE_LIMIT_CHAT_WINDOW",      str(_int(cp, "rate_limit", "chat_window", 60)))),
            chat_max         = int(_env("RATE_LIMIT_CHAT_MAX",         str(_int(cp, "rate_limit", "chat_max", 30)))),
            llm_quota_window = int(_env("RATE_LIMIT_LLM_QUOTA_WINDOW", str(_int(cp, "rate_limit", "llm_quota_window", 3600)))),
            llm_quota_max    = int(_env("RATE_LIMIT_LLM_QUOTA_MAX",    str(_int(cp, "rate_limit", "llm_quota_max", 100)))),
        )

        # ---- MySQL↔Milvus 数据对账 ----
        self.reconciliation = ReconciliationConfig(
            sync_interval   = int(_env("RECON_SYNC_INTERVAL", str(_int(cp, "reconciliation", "sync_interval", 3600)))),
            auto_clean      = _bool(cp, "reconciliation", "auto_clean", False),
            alert_threshold = _int(cp, "reconciliation", "alert_threshold", 10),
        )

        # ---- 启动校验：关键配置缺失时 fail-fast，避免首请求才崩 ----
        self._validate()

    def _validate(self) -> None:
        """校验关键配置项，缺失时抛出 ValueError 并给出明确提示。

        测试环境可通过环境变量 SKIP_CONFIG_VALIDATION=1 跳过校验。
        """
        if os.getenv("SKIP_CONFIG_VALIDATION", "").lower() in ("1", "true", "yes"):
            return

        errors: list[str] = []

        # LLM API Key 是生成回答的必要条件
        if not self.llm.api_key:
            errors.append(
                "LLM API Key 缺失：请设置环境变量 DASHSCOPE_API_KEY 或在 config.ini [llm] 段配置。"
            )

        # MySQL 数据库名不能为空
        if not self.mysql.database:
            errors.append("MySQL database 名称为空，请在 config.ini [mysql] 段配置 database。")

        # Milvus collection 名不能为空
        if not self.milvus.collection_name:
            errors.append("Milvus collection 名称为空，请在 config.ini [milvus] 段配置 collection_name。")

        # 本地模型路径必须存在（BGE-M3 / Reranker / BERT 是检索核心）
        for label, path in [
            ("BGE-M3", self.bge_m3_path),
            ("BGE-Reranker", self.bge_reranker_path),
        ]:
            if not Path(path).exists():
                errors.append(
                    f"{label} 模型路径不存在: {path}，请确认模型已下载。"
                )

        if errors:
            msg = "配置校验失败，请修正以下问题后重启:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)


# 全局单例
_instance: Optional[Config] = None
_instance_lock = threading.Lock()  # 双检锁：避免并发首请求重复构建配置


def get_config() -> Config:
    """获取全局配置单例（线程安全）。"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Config()
    return _instance


def reload_config() -> Config:
    """重新加载配置（测试用）。"""
    global _instance
    with _instance_lock:
        _instance = Config()
    return _instance
