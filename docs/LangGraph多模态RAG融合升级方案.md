# FinRag 融合升级方案 —— LangGraph 多模态 RAG（融合 ChatBI + DocAudit）

| 项目 | 内容 |
|---|---|
| 版本 | v1.0（三阶段已实施完成） |
| 日期 | 2026-09-07（方案）/ 2026-09-07（P1·P2·P3 全部落地） |
| 状态 | **✅ 已实施：P1（多模态审查）→ P2（NL2SQL）→ P3（LangGraph 编排）全部完成并通过验证** |
| 运行环境 | Windows 11 / 16G / PyCharm / uv（venv: D:\FinRag\.venv，Python 3.12） |
| 中间件 | MySQL 8、Redis 7、Milvus 2.4+（三项目共用实例，各自独立 database） |
| 宿主底座 | **FinRag**（认证/会话/知识库/流式/评测/治理已成熟，作为承载项目） |

---

## 1. 结论（可行性判断）

**可以融合，而且天然适合融合。** 依据有三：

1. 三个项目的架构文档已经把彼此写成了"双引擎/复用底座"：DocAudit 3.3 明确"检索直接复用 FinRag 混合检索代码，只写适配层"；ChatBI 9.2 明确"会话/限流/降级链复用 FinRag/OneThing"。三者共用同一套中间件与 `API_KEY`（百炼），依赖栈高度同构（FastAPI + sqlglot/openai + pymilvus + FlagEmbedding + BGE-M3/bge-reranker 本地权重路径完全一致）。**不存在"非要重写"的部分，主要是搬运 + 用 LangGraph 重新编排。**
2. FinRag 是三者中唯一**已具备成熟用户系统、会话隔离、管理员后台、RAGAS 评测闭环**的基础设施，最适合做"承载外壳"。
3. **LangGraph 的来源是 DocAudit**（`app/review/graph.py`，已用 StateGraph 实现 prepare→review_batch→…→report 的完整多 Agent 流水线，含条件边循环与 `astream` 节点级进度直通 SSE）。把这份 LangGraph 编排从"文档审查"这一条专用链路，升级为"多模态文档问答 + 结构化问数 + 文档审查"的统一编排层，即完成技术升级。

> 一句话回答用户问题：**能**。方案是把 FinRag 从"纯文档 RAG 问答"升级为"以 LangGraph 统一编排、同时具备【文档问答 RAG / 结构化数据 NL2SQL / 多模态文档解析审查】三种能力"的多模态 RAG 系统，DocAudit 提供多模态解析与 LangGraph 骨架，ChatBI 提供 NL2SQL 与 SQLGuard，FinRag 提供宿主与底座。

---

## 2. 三项目现状速览与归位

| 能力域 | FinRag（宿主） | ChatBI（融合来源） | DocAudit（融合来源） |
|---|---|---|---|
| 认证/会话/角色/后台 | ✅ 成熟 | ✖（仅简单 X-User） | ✖ |
| 文档 RAG 问答 | ✅ 混合检索+Rerank+BM25 | ✖ | ✖（但复用其检索模式） |
| 多模态文档解析 | ⚠️ 单文件 OCR 回退（`edu_ocr.py` rapidocr） | ✖ | ✅ **页级分级路由（digital/ocr/qwen-vl）** |
| 结构化条款抽取 | ✖ | ✖ | ✅ 抽取+三重反幻觉校验+重试 |
| NL2SQL 问数 | ✖ | ✅ SQLGuard 三层纵深 + 自修复 | ✖ |
| LangGraph 编排 | ✖ | ✖（自研调度器 chat_service） | ✅ **StateGraph 流水线** |
| HITL 人工复核 | ✖ | ✖ | ✅ 状态机（MySQL 持久化） |
| 评测闭环 | ✅ RAGAS（文档问答） | ✅ 执行准确率（NL2SQL） | ✅ CER/F1/召回（解析审查） |
| 治理层（审计/限流/metrics） | ✅ | ✅ | ✅ |

**核心搬运物：**
- 从 **DocAudit** 拿：`app/parsing/*`（多模态分级解析）、`app/review/graph.py`（LangGraph 骨架）、`app/extract/clause_agent.py`（结构化抽取）、`app/hitl/state.py`（HITL）、`app/retrieval/rule_retriever.py`（规则/条款检索）、`app/core/report_builder.py`（溯源报告）。
- 从 **ChatBI** 拿：`app/agents/nl2sql.py` + `app/agents/rewrite.py` + `app/agents/interpret.py` + `app/agents/repair.py`（NL2SQL Agent 组）、`app/guard/*`（sqlguard/executor/fingerprint/permission）、`app/retrieval/schema_retriever.py`、`app/services/metric_dict/chart/masking/normalize_time`、`app/governance/*`。
- **FinRag 承载**：`routers/auth/conversation/admin/knowledge_base`、`services/llm/embedding/reranker/bm25/intent`、`rag_qa/core/rag_system.py`、`services/evaluation.py`（RAGAS）。

---

## 3. 融合目标与原则

### 3.1 目标
把 FinRag 升级为一个 **LangGraph 统一编排的多模态 RAG 系统**，回答四类输入：

```
用户输入
 ├─ 非结构化问题（政策/研报/FAQ）     → RAG Skill（现有 FinRag 混合检索）
 ├─ 结构化问数（GMV/退货率/排名）      → NL2SQL Skill（ChatBI）
 ├─ 需要读图/扫描件才能回答的问题       → 多模态解析 Skill（DocAudit）+ RAG
 └─ 上传文档做合规审查                 → 文档审查 Workflow（DocAudit，LangGraph + HITL）
```

### 3.2 原则
1. **FinRag 为壳，能力为插**：不改动/不破坏现有 QA 链路已通过的评测。LangGraph 作为**新增编排层**，老链路（`/chat/stream` 直接管线）可通过配置开关保留，作为"自研调度 vs LangGraph"的对照（面试叙事要点）。
2. **复用不重写**：三个项目已互相复用，搬运单元按文件/模块整体迁入，只写适配层（配置、db 连接、models/__init__ restructure）。
3. **独立 database / collection 命名空间**：Milvus 增加 `chatbi_*`、`docaudit_*` 命名空间；MySQL 增加 `chatbi_meta`、`docaudit_meta` 库（或统一 `finrag_meta` 加命名前缀表），避免与 FinRag 现有 collection/表冲突。
4. **评测门禁不倒退**：三套评测各自保留；融合后新增"编排路由准确性"评测（判断问题该走哪条 Skill）。
5. **渐进式交付**：分成 3 个阶段，每阶段可独立跑通、可回归、可演示。

---

## 4. 融合后总体架构

```
                                ┌─────────────────────────────────────────────┐
                                │  ⑦ 治理层 F i n R a g 既有                 │
  用户                          │  audit_log│JWT+RBAC│Redis限流│/metrics│配额  │
   │                           └───────────────────┬─────────────────────────┘
   ▼                                             │
┌────────────────┐  路由(含新 /chat/ask 统一入口) │
│ FastAPI 接入层   │──────────┐                  │
│ /auth /kb /doc │          │ 鉴权/限流/审计     │
│ /conversation  │          ▼                   ▼
│ /admin /eval   │   ┌───────────────────────────────┐
└────────────────┘   │  ① 编排层 QueryGraph(LangGraph) │ ←核心升级
                     │  route 意图路由(条件边)         │
                     │  router → {rag | nl2sql |       │
                     │            multimodal | review} │
                     └───────┬────────┬────────┬───────┘
            ┌────────────────┘        │        └────────────────┐
            ▼                         ▼                        ▼
┌────────────────────┐   ┌────────────────────────┐   ┌──────────────────────┐
│ ② RAG Skill        │   │ ③ NL2SQL Skill(ChatBI) │   │ ④ 文档审查Workflow    │
│ 现有 FinRag 管线     │   │ Intent→Rewrite→Schema  │   │ (DocAudit LangGraph)  │
│ 意图→混合检索→Rerank │   │ →NL2SQL→SQLGuard→执行  │   │ 解析→抽取→审查→报告HITL│
│ →LLM               │   │ →自修复→解读→图表        │   │                      │
└─────────┬──────────┘   └─────────┬──────────────┘   └─────────┬────────────┘
          │                        │                             │
          └────────┬───────────────┘                             │
                   ▼                                             ▼
        ┌─────────────────────────┐          ┌───────────────────────────────┐
        │ ⑤ 多模态解析层(DocAudit) │          │ ⑧ 数据层                       │
        │ 数字页PyMuPDF/扫描RapidOCR│          │ MySQL: finrag_qa/chatbi_meta/ │
        │ /低置信qwen-vl兜底       │          │        docaudit_meta          │
        │ →结构化条款→向量入库      │          │ Milvus: finrag_faq/chatbi_* / │
        └─────────────────────────┘          │        docaudit_*              │
                                             │ Redis: 缓存/会话/进度/限流     │
                                             └───────────────────────────────┘
```

---

## 5. 核心：LangGraph 编排设计

新增 `rag_qa/orchestrator/graph.py`（LangGraph），承接 DocAudit `graph.py` 的骨架并扩展为三种入口。

### 5.1 统一问答图 `AnswerGraph`（新，核心升级）

```
State: { question, rewritten, intent, skill, domain, session_id,
         context, sources, sql, rows, table, chart, answer, trace, stage }

  route(意图路由，条件边)
    ├─ 命中文档关键词库/需检索 → ④ rag_retrieve → rag_answer → done
    ├─ 命中问数信号词(GMV/环比/排名/表) → ⑤ nl2sql →
    │      rewrite_time → schema_recall → sql_gen → sqlguard →
    │      ├─ GUARD_REJECT → repair(≤2轮) → 通过 → exec → interpret → done
    │      └─ REFUSE → friendly_refuse → done
    ├─ 命中"读图/扫描件"歧义 → ③ multimodal → 解析(§5.2) → 回调 rag_retrieve → done
    └─ chitchat/寒暄        → ② general_answer → done
```

节点与边（等价描述的 StateGraph）：

```python
g = StateGraph(AnswerState)
g.add_node("route", route)                     # 意图/技能路由（条件边）
g.add_node("rag_retrieve", rag_retrieve)       # 复用 RAGSystem.build_context
g.add_node("rag_answer", rag_answer)           # 复用 generate_answer
g.add_node("nl2sql", nl2sql_pipeline)          # 串起 ChatBI agents+guard
g.add_node("multimodal", multimodal_parse)     # DocAudit parsing router
g.add_node("general_answer", general_answer)
g.set_entry_point("route")
g.add_conditional_edges("route", decide_skill, {
    "rag": "rag_retrieve", "nl2sql": "nl2sql",
    "multimodal": "multimodal", "chitchat": "general_answer",
})
g.add_edge("rag_retrieve", "rag_answer")
g.add_edge("rag_answer", END)
g.add_edge("general_answer", END)
g.add_edge("nl2sql", END)                      # nl2sql 内部再串子链（可用 Subgraph）
g.add_edge("multimodal", "rag_retrieve")       # 解析结果向量入库后走 RAG 问答
g.compile()
```

> 复用方式：DocAudit `build_graph`/`run_review` 的 `astream(stream_mode="updates")` 节点级进度直通 SSE 的模式，直接搬进 `AnswerGraph` 的 `run` 包装里，保证 `/chat/stream` 前端零改造即可接收 `rag/nl2sql/table/chart/error/done` 事件。

### 5.2 多模态文档解析图 `DocumentIngestGraph`（融合 DocAudit parsing）

```
State: { file_path, file_hash, doc_type, pages[], units[], clauses[], status }

  detect_type(逐页探测文本层)
    〔条件边〕page → {digital | ocr}
  PyMuPDF_extract  ← digital
  RapidOCR_parse   ← ocr → (低置信条件边)→ qwen_vl_fallback
  clause_extract(三重校验+重试≤2)
  vectorize&store(父子分块 + BGE-M3 入库)
  └→ status=indexed
```
- **升级点**：FinRag 现有的 `rag_qa/core/document_processor.py` 是"整文件一把梭 + OCR 回退"，融合后替换为 DocAudit 的**页级分级路由**（数字页零成本直提，扫描页 RapidOCR，低置信页 qwen-vl 兜底），并保留 FinRag 的双层父子分块与 Milvus 入库。
- 上传接口 `POST /document/upload` 增加 `doc_type` 与"完成后自动入 RAG 索引"的选项，与现有知识库打通。

### 5.3 文档审查图 `DocReviewGraph`（DocAudit 原样迁入，作为独立 Workflow）

即 DocAudit `app/review/graph.py`（prepare→review_batch（检索+比对+分级）→条件边循环→report），迁入 `rag_qa/review/` 目录，依赖 FinRag 的检索底座（`db/milvus.py`、`services/embedding.py`）。HITL 状态机 (`pending_review→approved/rejected`) 复用 DocAudit `app/hitl/state.py`，MySQL 持久化。

---

## 6. 模块融入映射表（拿什么→放哪→改什么）

| 原项目文件 | 迁入 FinRag 位置 | 改动 |
|---|---|---|
| DocAudit `app/parsing/router.py` | `rag_qa/multimodal/router.py` | 读 FinRag config |
| DocAudit `app/parsing/pymupdf_extractor.py` | `rag_qa/multimodal/pymupdf_extractor.py` | 无 |
| DocAudit `app/parsing/paddle_parser.py` | `rag_qa/multimodal/ocr_parser.py` | 保持 rapidocr 默认，`DOCAUDIT_OCR_ENGINE`→`FINRAG_OCR_ENGINE` |
| DocAudit `app/parsing/vl_fallback.py` | `rag_qa/multimodal/vl_fallback.py` | 读 FinRag `services/llm.py` 的 async client |
| DocAudit `app/extract/clause_agent.py` | `rag_qa/multimodal/clause_extract.py` | 复用 `services/llm.py` |
| DocAudit `app/review/graph.py` | `rag_qa/review/graph.py` | 依赖改指 FinRag db/services |
| DocAudit `app/hitl/state.py` | `rag_qa/review/hitl.py` | 无 |
| DocAudit `app/retrieval/rule_retriever.py` | `rag_qa/core/rule_retriever.py` | 复用 `services/embedding.py` `db/milvus.py` |
| ChatBI `app/agents/nl2sql.py / rewrite.py / interpret.py / repair.py / intent.py` | `rag_qa/nl2sql/agents/*.py` | 复用 `services/llm.py` 的 `chat_json`（需在 services/llm.py 补一个 `chat_json_validated` 包装） |
| ChatBI `app/guard/sqlguard.py / executor.py / fingerprint.py` | `rag_qa/nl2sql/guard/*.py` | 只读连接池改复用 FinRag `db/mysql.py` |
| ChatBI `app/retrieval/schema_retriever.py` | `rag_qa/nl2sql/schema_retriever.py` | Milvus 用 FinRag client |
| ChatBI `app/services/chart.py / masking.py / metric_dict.py / normalize_time.py` | `services/nl2sql_proc/*.py` | 无 |
| FinRag 现有 `routers/chat.py` | 保留 + 新增 `/chat/ask`（走 AnswerGraph） | 老路径开关保留 |
| — | 新增 `rag_qa/orchestrator/graph.py + state.py + run.py` | **核心新增** |

**编排层新增文件（本项目自研，勿搬现成）：**
- `rag_qa/orchestrator/state.py` — 统一 State TypedDict
- `rag_qa/orchestrator/graph.py` — AnswerGraph + 条件边
- `rag_qa/orchestrator/route.py` — 意图→Skill 路由（LLM 三分类 + 关键词先行，复用 FinRag 四分类 BERT 兜底）
- `rag_qa/orchestrator/progress.py` — `astream(updates)` 节点→SSE 事件桥

---

## 7. 数据与 Schema 变更

- **MySQL**：三 model 各自的 `chatbi_meta`、`docaudit_meta` 库随迁入一起建（对应原 DDL）；FinRag `finrag_qa` 不变。新增 `llm_trace`（委员会价统一 trace_id 贯穿三 Skill）。
- **Milvus**：新增 collection 命名空间 `chatbi_schema_vectors / chatbi_fewshot_vectors / chatbi_qa_cache_vectors / chatbi_metric_vectors / docaudit_rule_vectors / docaudit_clause_vectors`；FinRag `finrag_faq` 不变。全部用 FinRag 的 BGE-M3 稠密+稀疏 schema（若引入云向量通道则按 ChatBI 维度自动探测，融合层统一 `embedding_dim`）。
- **config.ini**：新增 `[orchestrator]`（enable_answer_graph、route_fallback）、`[nl2sql]`（sql_model/fast_model/max_limit/repair_rounds）、`[multimodal]`（digital_min_chars/ocr_confidence_route/batch_size）。

---

## 8. 依赖变更与内存预算

依赖增量（3 项目去重后仅新增少量）：
```
langgraph>=1.2        # DocAudit 已有
rapidocr-onnxruntime  # FinRag 已有
pymupdf               # FinRag 已有
sqlglot               # ChatBI 已有，FinRag 需新增
python-dotenv         # 若复用 ChatBI config
jinja2                # DocAudit 报告模板
```

| 组件 | 预估内存（16G） |
|---|---|
| FinRag 应用（含 BGE-M3/reranker/BERT） | ~4.5G |
| Milvus standalone | ~3G |
| MySQL / Redis | ~0.8G |
| RapidOCR + LangGraph worker | ~0.5G |
| **合计** | **< 9G，余量充足**（99% 解析走本地，qwen-vl 按页云计费） |

---

## 9. 分阶段实施路线（每阶段可跑通可回归可演示）

| 阶段 | 交付 | 验收门禁 | 状态 |
|---|---|---|---|
| **P1：搬 DocAudit → 多模态文书工程** | `rag_qa/multimodal/*` + `rag_qa/review/*`；文档上传升级为页级分级解析；审查 Workflow + HITL；`/document/review` 接口 | 数字PDF≥98%、扫描1-CER≥95%、抽取F1≥90%、审查召回≥85%（复用 DocAudit 评测集）；FinRag 现有 RAGAS 评测不倒退 | ✅ 完成：`routers/review.py` 7 接口 + `rag_qa/review/*`，LangGraph 审查链路冒烟通过（解析→抽取→审查→completed），LLM 失败优雅降级为 insufficient_evidence |
| **P2：搬 ChatBI → NL2SQL Skill** | `rag_qa/nl2sql/*`；SQLGuard 全量单测；`/chat/ask` 初步支持问数；few-shot/metadata 入库 | SQLGuard 通过率≥97%、陷阱题拦截100%、简单题执行准确率≥95% | ✅ 完成：`rag_qa/nl2sql/*` + `routers/nl2sql.py`；qwen3.7-flash 真实 LLM 端到端问数成功，SQLGuard 9/9 攻击向量全拦截，只读账号拒写通过 |
| **P3：LangGraph 编排与评测收口** | `rag_qa/orchestrator/*` 统一 `AnswerGraph`；`/chat/ask` 全 Skill 路由；新增"路由准确性"评测集（200 题标 skill 标签）；README/方案文档更新 | 三 Skill 各自评测指标全绿 + 路由准确率≥90%；新旧路径对照可演示 | ✅ 完成：`rag_qa/orchestrator/{state,route,nodes,graph}.py` + `routers/chat.py` 新增 `/chat/ask`（统一）+ `/chat/ask_stream`（SSE）；路由评测 218/218=100%（门禁90%） |

> 每阶段都跑 `pytest -q`（FinRag 现有 20+ 测试文件）防止回归；ChatBI/DocAudit 的 `tests/` 迁入后并入，单测总数预计 250+。

---

## 10. 风险与规避

| 风险 | 规避 |
|---|---|
| 搬运破坏 FinRag 现有 QA 链路 | LangGraph 作为增量编排层；老路径配置开关保留；每阶段跑全量 pytest + RAGAS 对比 |
| 三套 config/db 连接各自为政 | 统一收口到 FinRag `config.py` + `db/mysql.py` + `db/milvus.py`（第三个项目已证明可复用，只写适配层） |
| OCR 中文路径 / Windows 崩溃 | 已采 RapidOCR 自包含引擎 + `PADDLE_PDX_CACHE_HOME` ASCII 路径（DocAudit 踩坑实录直接沿用） |
| 对话模型 thinking 超时 | `enable_thinking=False`（确定性任务），Deg-链复用 FinRag `services/llm.py` 降级 |
| NL2SQL 安全 | 只读账号 + SQLGuard 三层纵深原样迁入，表名白名单硬校验防幻觉表名 |
| 评测指标混合导致门禁失真 | 三套评测相互独立，各自门禁；新增"路由准确性"仅作为编排层附加门禁 |
| 代理劫持本地请求 | 全链路 `trust_env=False` + NO_PROXY（三个项目一致） |

---

## 11. 面试叙事升级点

1. **一题双作答 / 三引擎故事**：同一自然语言入口，LangGraph 条件边路由到"文档 RAG / NL2SQL / 多模态解析"——比三个独立 Demo 更能体现架构能力；
2. **两种编排范式对比**：LangGraph（固定流程图+条件分支+checkpoint，适合审查/问答路由）vs 自研调度器（FinRag 老链路 / ChatBI `chat_service`，适合深度定制的治理语义）——可讲适用边界；
3. **多模态分级解析的成本-精度-延迟权衡**：数字页零成本直提、扫描页本地 RapidOCR、疑难页 qwen-vl 兜底，给解析器分布 metrics；
4. **反幻觉三件套 + SQLGuard 三层纵深**：抽取侧页码回验/编号连续性/内容 diff，审查侧强制 rule_id+页码溯源（代码校验），NL2SQL 侧 AST 校验+EXPLAIN+只读执行；
5. **评测驱动可复现**：RAGAS（文档问答）+ 执行准确率（NL2SQL）+ CER/F1/召回（解析审查）+ 路由准确率（编排），四套门禁全部脚本化留痕。

---

## 12. 结论

**可行，且已按 P1→P2→P3 全部落地。** P1（DocAudit：多模态解析 + LangGraph 审查）、P2（ChatBI：NL2SQL + SQLGuard）、P3（LangGraph 统一编排 `AnswerGraph` + 路由评测）均已实现并通过验证，设计稿 → 可运行系统的闭环已完成。

---

## 13. 实施实录（P1—P3）

### 13.1 落地产物

| 阶段 | 新增/改动文件 | 关键点 |
|---|---|---|
| P1 多模态审查 | `rag_qa/review/*`、`routers/review.py`、`main.py` 挂载 7 审查接口 | 幂等建表；LangGraph 审查；（抽取/审查链路冒烟：解析→抽取→审查→completed） |
| P2 NL2SQL | `rag_qa/nl2sql/*`（service/executor/sqlguard/schema_retriever/embeddings/generation/meta_store/schema/prompts/db）、`routers/nl2sql.py`、`config.ini[llm]/[nl2sql]` | qwen3.7-flash 生成 + qwen3.7-text-embedding-flash 召回；SQLGuard 三层纵深；只读账号 `chatbi_ro` |
| P3 统一编排 | `rag_qa/orchestrator/{state,route,nodes,graph}.py`、`routers/chat.py` 新增 `/chat/ask` + `/chat/ask_stream`、`config.ini[orchestrator]` | AnswerGraph 条件边路由；懒加载避循环依赖；`tests/test_orchestrator.py` 单测 |
| P3 评测 | `scripts/route_skill_eval.json`（218 题，标 skill）、`scripts/eval_routing.py`（门禁 ≥90%） | 纯规则离线可复现，退出码反映门禁 |

### 13.2 运行与验证

```powershell
# 启动（需先设置环境变量；api_key 从 DASHSCOPE_API_KEY / API_KEY 注入）
$env:API_KEY="<你的百炼Key>"
D:\FinRag\.venv\Scripts\python.exe main.py

# 统一编排入口（完整链路：RAG / NL2SQL / chitchat 自动路由）
POST /api/v1/chat/ask        { "message": "华东区近7天销售额是多少?" }
POST /api/v1/chat/ask_stream # SSE，节点级进度直通
# NL2SQL 直查 / 详情
POST /api/v1/nl2sql/ask      POST /api/v1/nl2sql/execute
GET  /api/v1/nl2sql/schema   POST /api/v1/nl2sql/guard_check
# 多模态审查
POST /api/v1/review/upload

# 路由准确率评测（门禁 90%）
D:\FinRag\.venv\Scripts\python.exe scripts/eval_routing.py
# 编排层单测
D:\FinRag\.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py -q
```

### 13.3 实施中踩过的坑（记录）

1. **`route_node` 必须返回增量 dict**——LangGraph 只合并节点返回值，仅原地改 state 不会传给条件边，会导致始终路由到默认 rag。
2. **条件边 read-after-write**——用 `_branch(state)` 读 route 节点写回的 `skill` 做分发，multimodal 归并到 rag 检索边。
3. **循环依赖**——orchestrator.nodes 复用 `routers.chat` 的检索/降级逻辑；`/chat/ask` 引入 orchestrator 改为函数内局部 import，避免 `routers.chat ↔ orchestrator.nodes` 循环。
4. **中文引号/ASCII 引号冲突**——NL2SQL 服务提示文案里中文引号误用 ASCII `"`，导致整个模块无法编译；改为单引号拼接。
5. **dict 游标返回值是列名**——executor 用 `for row in cur.fetchall()` 迭代到的是表头而非行值，改 `.values()` 取真实数据。
6. **`_format_result` 生成器作用域**——样本行在生成器表达式内未定义，改为先取 `rows[0]`。
7. **Embedding 批次限制**——百炼单次入参批大小有限制，按 batch=5 分批调用再拼接；`dimensions` 参数不被支持时降级去参重试。
8. **单字信号词过宽**——路由词表里的 `客户/费用/净/前` 等导致政策类问题误路由 NL2SQL，全部去除。

### 13.4 待办 / 归口说明

- 路由评测集 `route_skill_eval.json` 目前按既定词表标注（自标），验证的是"规则实现与标注一致 + 可复现"；**更硬的门禁**是把这 218 题换成外部/人工独立标注后再跑 `eval_routing.py`。
- langsmith/langgraph-checkpoint 持久化、`progress.py` 独立幕墙未单列——SSE 节点级进度已并入 `graph.stream_ask`，如需 checkpoint/HITL 分支可在此之上扩展。
- P2 的 few-shot/metadata 向量入库为按需初始化（`MetaStore` 首用自动建表），未做批量预建脚本。