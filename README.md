# FinRag — 金融领域智能问答系统

基于 RAG（检索增强生成）的企业级金融智能问答系统，集成 BM25 稀疏检索、Milvus 向量检索、BGE-Reranker 重排序以及微调 BERT 四分类意图识别，支持在线流式对话、知识库管理、用户角色权限体系、会话隔离、后台管理面板以及基于 RAGAS 框架的 RAG 系统评估。用户会话按 user_id 严格隔离，管理员可查看所有用户的历史对话并管理账号；RAGAS 评估通过 LLM 自动生成评估数据集，量化评估检索质量与回答准确性。

---

## 目录

1. [系统架构](#系统架构)
2. [技术栈](#技术栈)
3. [快速开始](#快速开始)
4. [模块说明](#模块说明)
5. [API 接口](#api-接口)
6. [数据存储设计](#数据存储设计)
7. [前端界面](#前端界面)
8. [测试](#测试)
9. [配置说明](#配置说明)
10. [目录结构](#目录结构)

---

## 多模态融合升级（P1 / P2 / P3）

在原 RAG 文本问答之上，本项目已融合另外两个项目的能力，升级为「**多模态 + 结构化问数 + 统一编排**」的企业级智能问答系统。三个阶段的详细设计与落地实录见 [docs/LangGraph多模态RAG融合升级方案.md](docs/LangGraph多模态RAG融合升级方案.md)。

| 阶段 | 融合来源 | 能力 | 落地位置 |
|------|---------|------|---------|
| **P1** | DocAudit | 多模态文档解析 + LangGraph 合规审查（抽取→审查→人工复核 HITL） | `rag_qa/multimodal`、`rag_qa/review`、`routers/review.py` |
| **P2** | ChatBI | 自然语言→SQL（NL2SQL）+ SQLGuard 三层纵深防御 + 只读执行 | `rag_qa/nl2sql`、`routers/nl2sql.py` |
| **P3** | 统一编排 | LangGraph AnswerGraph 路由到 rag / nl2sql / chitchat，含路由准确率评测门禁 | `rag_qa/orchestrator`、`routers/chat.py` |

> 融合后功能开关均在 `config.ini`：`[orchestrator] enable_answer_graph` 关闭时回退老链路。

---

## 系统架构

```
用户 Query
    │
    ▼
┌─────────────────── LangGraph AnswerGraph（统一编排 / after P3）───────────────────┐
│  路由 Skill：multimodal > nl2sql > chitchat > rag（关键词策略 + 可选 LLM 三分类）  │
│                                  │                                                 │
│              ┌───────────────────┼───────────────────┐                             │
│              ▼                   ▼                   ▼                             │
│          rag_retrieve/      nl2sql（P2）          chitchat（闲谈）                 │
│          rag_answer             │                      │                           │
└──────────────┬──────────────────┼──────────────────────┘                           │
               │                  ▼                                                  │
               │        SQLGuard 三层防御：AST 校验/规格化 → EXPLAIN 预检 →             │
               │        只读账号 + READ ONLY + LIMIT/超时                              │
               ▼                  (biz_demo 只读执行)                                  │
       结构化问答 / 知识检索 / 多模态审查 输出
    │
    ▼
┌──────────────┐
│  Redis 缓存   │── 命中 → 直接返回答案
└──────┬───────┘
       │ 未命中
       ▼
┌──────────────┐
│  BM25 高频匹配│── 相似度 > 阈值 → 返回 MySQL 答案
└──────┬───────┘
       │ 未命中
       ▼
┌──────────────────┐
│  BERT 意图识别    │── 四分类：banking / corporate_finance
│  (微调模型)       │         / financial_accounting / general
└────────┬─────────┘
         │ 金融领域
         ▼
┌─────────────────────┐
│  检索策略路由        │── 规则预筛（零开销）：
│  (multi-strategy)   │   直接检索 / HyDE 改写 / 子问题分解 / 回溯简化
└──────────┬──────────┘   （未命中信号词一律直接检索）
           │
           ▼
┌─────────────────────┐
│  混合召回            │
│  • 稠密向量 (BGE-M3) │
│  • 稀疏向量 (BGE-M3) │
│  • BM25 (FAQ)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Reranker 精排       │
│  (BGE-Reranker-V2)  │
└──────────┬──────────┘
           │
           ▼
┌──────────────┐
│  LLM 生成     │── 流式返回 (SSE)
└──────────────┘
```

---

## 技术栈

| 层级 | 组件 | 技术选型 |
|------|------|---------|
| 后端框架 | FastAPI + Uvicorn | 异步高性能，自动生成 OpenAPI 文档 |
| 向量数据库 | Milvus 2.4.x | 稠密+稀疏双向量混合检索 |
| 关系数据库 | MySQL 8.0 | 用户、知识库、文档、FAQ 记录 |
| 缓存 | Redis 7.0 | 问答缓存、BM25 分词数据、会话历史、验证码 |
| 嵌入模型 | BGE-M3 | 1024 维稠密向量 + 稀疏向量 |
| 重排序 | BGE-Reranker-V2-M3 | 精排召回候选 |
| 意图识别 | 微调 BERT (bert-base-chinese) | 四分类：银行/公司金融/财务会计/通用 |
| 检索 | rank_bm25 + jieba + 金融词典 | 本地 BM25 高频匹配 |
| 大模型 | 通义千问 qwen3.7-flash (DashScope API) | 流式生成回答、NL2SQL、编码重写、意图精判 |
| Embedding | qwen3.7-text-embedding-flash（阿里云百炼 API） | NL2SQL Schema 向量召回 |
| 编排 | LangGraph (StateGraph + 条件边) | AnswerGraph 统一路由 rag / nl2sql / chitchat |
| SQL 安全 | SQLGuard（sqlglot AST 校验） | 结构化问数三层纵深防御 + 只读执行 |
| 文档解析 | langchain + pymupdf + python-docx | PDF / DOCX / Markdown / TXT |
| 认证 | JWT + bcrypt | Token 认证、密码哈希存储、角色权限控制 |
| 前端 | HTML + CSS + JavaScript | 单页应用，ChatGPT 风格界面 |
| 包管理 | uv | 快速依赖管理 |
| RAG 评估 | RAGAS | RAG 系统评估（忠实度、上下文召回、事实正确性、回答相关性） |
| 测试框架 | pytest | 单元测试 + 集成测试 |

---

## 快速开始

### 环境要求

- Python 3.12+
- MySQL 8.0（数据库：`finrag_qa`）
- Redis 7.0
- Milvus 2.4.x（数据库：`finrag`，collection：`finrag_faq`）

### 安装

```bash
# 进入项目目录
cd D:\FinRag

# 创建虚拟环境（若未创建）
uv venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 数据库初始化

```bash
# 创建数据库和表
mysql -u root -p < init_sql/init.sql

# 初始化 users 表（含 role/email/avatar/status 字段，创建默认管理员）
.venv\Scripts\python.exe create_users_table.py

# 导入内置数据（FAQ + 知识文档）
.venv\Scripts\python.exe init_data.py
```

### 启动服务

```bash
# 启动 FastAPI 服务
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

访问以下地址：

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/` | 前端页面（登录 / 注册 / 问答 / 知识库 / 管理面板） |
| `http://localhost:8000/docs` | Swagger API 文档 |
| `http://localhost:8000/redoc` | ReDoc API 文档 |
| `http://localhost:8000/health` | 健康检查 |

### 默认账号

| 角色 | 用户名 | 密码 | 说明 |
|------|--------|------|------|
| 管理员 | `admin` | `admin123` | 超级管理员，拥有全部权限 |
| 普通用户 | — | — | 通过注册页面创建 |

### RAG 评估

```bash
# 安装 RAGAS 依赖
cd D:\FinRag
.venv\Scripts\python.exe -m pip install ragas

# 一键运行完整评估（生成数据集 → RAG 管线 → RAGAS 评估）
# 方式一：通过 API（需管理员 Token）
curl -X POST http://localhost:8000/evaluation/run \
  -H "Authorization: Bearer <YOUR_ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"num_seeds": 10, "questions_per_seed": 3}'

# 方式二：通过 Python 脚本
.venv\Scripts\python.exe -c "
import sys; sys.path.insert(0, '.')
from services.evaluation import run_full_evaluation
results = run_full_evaluation(num_seeds=10, questions_per_seed=3)
print(results)
"
```

评估指标说明：

| 指标 | 说明 | 取值范围 |
|------|------|---------|
| Faithfulness | 回答忠实于检索上下文的程度，不编造信息 | 0-1（越高越好） |
| LLMContextRecall | 检索上下文覆盖参考答案关键信息的比例 | 0-1（越高越好） |
| FactualCorrectness | 回答与参考答案的事实一致性 | 0-1（越高越好） |
| AnswerRelevancy | 回答与用户问题的相关程度 | 0-1（越高越好） |

---

## 容器部署与生产建议

项目提供 `docker-compose.yml`，但**默认只启动应用容器**，通过 `host.docker.internal` 连接你本机已经运行的 Redis / MySQL / Milvus 服务。这样不会在 16G 内存的机器上同时跑两套中间件。

```bash
# 1) 准备配置与密钥
cp config.ini.example config.ini        # 按需修改（config.ini 已被 .gitignore 忽略）
cp .env.example .env                     # 填入 DASHSCOPE_API_KEY 等
# 确保本机 Redis / MySQL / Milvus 已启动

# 2) 构建并启动（仅 app 容器）
docker compose up -d --build

# 3) 查看健康状态
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

若是在**没有这些中间件的新环境**，用 `infra` profile 一并拉起：

```bash
docker compose --profile infra up -d --build
```

> **内存预算（16G 主机参考）**：宿主机原生 Milvus standalone 约 2~4G、MySQL ~0.5G、Redis ~0.2G；app 容器加载 BGE-M3(~2.3G) + Reranker(~1.1G) + BERT(~0.4G) 约 4~5G。总体约 8~10G，留有冗余。各服务在 compose 中已配置 `mem_limit` 防止失控。本地模型权重通过 volume 挂载进容器，不打包进镜像（见 `.dockerignore`），保持镜像精简。

### 安全与运维要点

- **密钥管理**：LLM API Key 仅通过环境变量（`.env` 或 `DASHSCOPE_API_KEY`）注入，绝不写入 `config.ini`；`config.ini`、`.env`、`.jwt_secret` 均已被 `.gitignore` 忽略。
- **跨域（CORS）**：生产环境请在 `config.ini` 的 `[app] cors_origins` 或环境变量 `CORS_ORIGINS` 中显式配置前端来源，**禁止 `*` 与凭据同时使用**。
- **连接管理**：MySQL 使用连接池（DBUtils），Redis / Milvus 使用进程级单例客户端，避免连接泄漏；服务关闭时通过 lifespan 优雅释放。
- **限流**：`/auth/login`、`/auth/register` 内置基础内存限流（60s 窗口 20 次），防爆破。
- **探针**：`/health` 用于存活探针，`/health/ready` 用于就绪探针（探活三大依赖）。
- **Schema 保护**：Milvus collection schema 不匹配时拒绝自动 drop（防清空数据），需人工运行 `scripts/recreate_collection.py --yes` 重建；开发环境可设 `MILVUS_AUTO_RECREATE=1`。
- **LLM 降级链**：配置 `fallback_model`（或环境变量 `LLM_FALLBACK_MODEL`）后，主模型失败自动用备用模型重试一次。
- **多 worker 部署**：设置 `PROMETHEUS_MULTIPROC_DIR` 后 `/metrics` 自动切换为多进程聚合采集。
- **会话存储**：会话消息通过 Lua 脚本原子追加（并发不丢消息），会话列表走 `finrag:user_sessions:{uid}` ZSET 索引（存量数据自动回填索引）。

## 模块说明

### 4.1 用户认证与权限模块

- **注册**：用户名 + 密码 + 确认密码 + 图形验证码，支持邮箱字段
- **登录**：用户名 + 密码 + 图形验证码，登录后签发 JWT（含 user_id 和 role）
- **Token 认证**：JWT Bearer Token，有效期 24 小时
- **JWT Secret 持久化**：Secret 存储于 `.jwt_secret` 文件，服务重启后旧 Token 仍然有效，实现自动登录
- **自动登录**：前端页面加载时自动检测 localStorage 中的 Token，验证有效则直接进入应用
- **密码安全**：bcrypt 哈希存储，不存明文
- **角色权限**：`admin`（管理员）和 `user`（普通用户）两种角色
  - 管理员可访问 `/admin/*` 接口，管理用户、查看系统统计、查看所有用户会话
  - 普通用户仅能使用问答和知识库功能，只能看到自己的会话
  - `get_current_user` 依赖注入：验证 Token 并返回用户信息
  - `get_admin_user` 依赖注入：验证管理员权限
- **验证码**：PIL 生成图形验证码，Redis 存储（5 分钟有效期）
- 路由：`/auth/captcha`、`/auth/register`、`/auth/login`、`/auth/me`

### 4.2 管理员后台模块

- **系统统计**：总用户数、管理员数、知识库数、文档数、FAQ 数
- **用户管理**：
  - 分页查询用户列表（支持用户名/邮箱搜索）
  - 更新用户角色（admin ↔ user）、状态（active ↔ disabled）、邮箱
  - 删除用户（不能删除管理员账号），删除时自动清理该用户的所有 Redis 会话
  - 保护超级管理员：不能修改 admin 的角色、不能删除 admin
- **用户会话查看**：
  - 查看任意用户的所有会话列表（标题、消息数、最后活跃时间）
  - 查看任意用户指定会话的完整对话历史
- 路由：`/admin/stats`、`/admin/users`、`/admin/users/{user_id}/sessions`、`/admin/users/{user_id}/sessions/{session_id}/history`

### 4.3 知识库管理模块

- **知识库 CRUD**：创建、删除、查询知识库（内置/自定义）
- **属主权限模型**：自定义知识库记录创建者（owner_id），写操作（改名/删除/上传/删除文档）仅属主与管理员可执行；内置知识库属主为系统，普通用户只读，管理员可写。应用启动时自动执行 owner_id 列的幂等迁移
- **文档上传**：支持 PDF、DOCX、Markdown、TXT，自动解析并分块；上传领域标签未指定时自动回退知识库领域
- **父子分块**：父块存储上下文，子块用于向量检索
- 路由：`/knowledge_base/`、`/document/`

### 4.4 高频问答对管理

- MySQL `finance_faq` 表存储标准问答对（含领域标签）
- 服务启动时加载至 Redis，构建本地 BM25 索引
- 命中阈值（默认 0.75）以上直接返回，否则进入 RAG 流程

### 4.5 在线检索与问答

- **多策略检索路由**：规则预筛（对比类→子查询分解、抽象开放类→HyDE 假设检索、场景复杂类→回溯简化），未命中信号词一律走直接检索（零额外开销）；改写只影响召回，精排始终以用户原始问题为准，改写失败自动降级。可在配置中关闭或开启 LLM 选择器兜底
- **四分类意图识别**：微调 BERT 模型，识别问题所属金融领域
- **混合检索**：BGE-M3 稠密向量 + BM25 稀疏向量，RRF 融合
- **Reranker**：BGE-Reranker-V2-M3 精排 Top-K
- **流式输出**：SSE 协议，LLM 回答实时流式返回
- **自动持久化**：流式接口内部自动保存用户消息和助手回答，无需前端额外调用

### 4.6 多轮对话

- 会话历史存储于 Redis（7 天 TTL，保留最近 50 条）
- **会话按 user_id 隔离**：每个会话绑定创建者的 user_id，用户只能看到自己的会话列表和历史
- **权限控制**：所有会话操作（保存/查询/清空/列表/新建）均需 Token 认证，非本人会话返回 403
- 会话标题自动提取：以用户首次提问的前 30 字作为标题
- 支持多轮上下文构建、历史轮数限制
- 管理员可通过 `/admin/users/{user_id}/sessions` 接口查看任意用户的会话
- 路由：`/conversation/save`、`/conversation/history`、`/conversation/clear`、`/conversation/list`、`/conversation/new`

### 4.7 RAG 系统评估（RAGAS）

基于 [RAGAS](https://docs.ragas.io/) 框架，通过 LLM 自动构建评估数据集，量化评估 RAG 管线的检索质量与回答准确性。

- **评估数据集生成**：
  - 从 MySQL `finance_faq` 表和本地 JSONL 数据文件中采样 FAQ 作为种子
  - 使用 LLM 对每条种子生成 3 种类型的评估问题：事实性、推理型、变换表述
  - 自动生成参考答案，标注领域类别
  - 数据集保存至 `evals/datasets/` 目录
- **RAG 管线收集**：
  - 对每个评估问题运行完整 RAG 管线（意图识别 → Milvus + BM25 混合检索 → Reranker → LLM 生成）
  - 收集 `user_input`、`retrieved_contexts`、`response`、`reference` 四元组
- **RAGAS 评估指标**：
  - **Faithfulness（忠实度）**：回答是否忠实于检索到的上下文，不编造信息
  - **LLMContextRecall（上下文召回率）**：检索到的上下文是否覆盖了参考答案的关键信息
  - **FactualCorrectness（事实正确性）**：回答与参考答案的事实一致性
  - **AnswerRelevancy（回答相关性）**：回答与用户问题的相关程度
- **评估结果**：
  - 汇总指标均值 + 每条样本的详细分数
  - 保存至 `evals/experiments/` 目录，支持历史查询
- **一键评估**：`POST /evaluation/run` 自动完成数据集生成 → RAG 收集 → RAGAS 评估全流程
- **实验对比闭环**：`GET /evaluation/experiments/compare`（或管理面板「RAG 评估对比」）逐指标对比多次实验的取值/最优/极差，支撑「调检索参数或 Prompt → 跑评估 → 对比」的效果迭代
- 路由：`/evaluation/generate-dataset`、`/evaluation/run`、`/evaluation/evaluate`、`/evaluation/datasets`、`/evaluation/experiments`、`/evaluation/experiments/compare`

### 4.8 多模态合规审查（融合 DocAudit，P1）

将 DocAudit 的「多模态分级解析 + LangGraph 审查」能力并入，用于监管/合规类文档的风险条条款审查。

- **多模态分级解析**：优先取 PDF 文本层，文本不足按配置走 OCR；OCR 置信度不足再用视觉模型兜底，页级并行
- **条条款抽取**：LLM 结构化抽取（条款编号/标题/内容/页码回验），抽取失败自动重试
- **LangGraph 多 Agent 审查**：逐条比对内置合规规则，输出违规条款 + 判罚依据；LLM 失败优雅降级为 `insufficient_evidence`
- **人工复核（HITL）**：自动审查结果可逐条 `apply` 采纳，支持回滚继续重审
- **规则库**：服务启动时幂等加载内置合规规则种子（`compliance_rule`）
- 生命周期用自增任务表驱动，阶段事件（parsing → extracting → reviewing → review_done / hitl_pending）通过 `events` 接口异步拉取
- 路由：`/review/upload`、`/review/task/{id}/status|events|clauses|reviews|report`、`/review/reviews/{id}/apply`、`/review/rules`

### 4.9 结构化问数（NL2SQL + SQLGuard，融合 ChatBI，P2）

将 ChatBI 的「自然语言 → SQL」能力并入，让用户用自然语言直接查询业务库数据，全程只读。

- **SQL 生成**：qwen3.7-flash 生成 SQL，失败时按 `repair_max_rounds` 自修复重写
- **Schema 召回**：关键词 + 向量（qwen3.7-text-embedding-flash）混合召回表/字段元数据，关键词命中表优先入上下文
- **SQLGuard 三层纵深防御**：
  1. **AST 校验**（sqlglot）：仅允许单条 `SELECT`，拦截多语句、DDL/DML、`SLEEP`、`INTO OUTFILE`、版本注释绕过、`FOR UPDATE` 等注入
  2. **EXPLAIN 预检**：预估扫描行数超阈值（默认 200 万）拒绝
  3. **只读执行**：`biz_demo` 通过只读账号 `chatbi_ro` 访问，会话级 `READ ONLY` + 强制 `LIMIT` + 查询超时
- **执行结果**：返回分类行数据 + 结构版表格（供前端渲染）+ 自然语言摘要
- 路由：`/nl2sql/ask`、`/nl2sql/execute`、`/nl2sql/schema`、`/nl2sql/guard_check`

### 4.10 统一编排（LangGraph AnswerGraph，P3）

用 LangGraph `StateGraph` 把多路 Skill 收敛到一张图，`/chat/ask` 一个入口即可命中 rag / nl2sql / chitchat。

- **AnswerState**：question / skill / context / sources / sql / table / answer / trace 统一状态，收敛四类输入
- **多 Skill 路由**：关键词策略先行（`multimodal > nl2sql > chitchat > rag`），规则未命中可开 LLM 三分类精判
- **节点**：`rag_retrieve`（复用老链路混合检索+降级）、`nl2sql`（复用 P2 服务）、`chitchat`、`route`
- **双出口**：`/chat/ask`（非流式，返回 skill/sources/table/sql/trace）+ `/chat/ask_stream`（SSE 节点级进度直通）
- **可回退**：`[orchestrator] enable_answer_graph=false` 时 `/chat/*` 走老链路
- **路由评测**：`scripts/eval_routing.py` + `scripts/route_skill_eval.json`（218 题带 Skill 标注）输出总准确率、各 Skill P/R 与混淆矩阵，默认门禁 90%

---

## API 接口

> 所有 API 路由统一挂载在 `/api/v1` 前缀下（便于版本演进），下表省略该前缀。
> 例如登录接口实际路径为 `POST /api/v1/auth/login`。

### 认证

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | `/auth/captcha` | 获取图形验证码（返回 captcha_id + base64 图片） | 无 |
| POST | `/auth/register` | 用户注册（用户名、密码、确认密码、验证码、邮箱） | 无 |
| POST | `/auth/login` | 用户登录，返回 JWT + 用户名 + 角色 | 无 |
| GET | `/auth/me` | 获取当前用户信息（验证 Token） | Bearer Token |

### 管理员

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | `/admin/stats` | 获取系统统计（用户数、知识库数、文档数等） | 管理员 |
| GET | `/admin/users?page=1&page_size=20&search=` | 分页获取用户列表 | 管理员 |
| PUT | `/admin/users/{user_id}` | 更新用户角色/状态/邮箱 | 管理员 |
| DELETE | `/admin/users/{user_id}` | 删除用户（同时清理其所有会话，不能删管理员） | 管理员 |
| GET | `/admin/users/{user_id}/sessions` | 获取指定用户的会话列表 | 管理员 |
| GET | `/admin/users/{user_id}/sessions/{session_id}/history` | 获取指定用户的会话历史 | 管理员 |

### 问答

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/chat/` | 问答接口（非流式） | Bearer Token |
| POST | `/chat/stream` | 问答接口（SSE 流式），自动保存对话消息 | Bearer Token |
| POST | `/chat/ask` | 统一编排问答（LangGraph AnswerGraph 全 Skill 路由，非流式） | Bearer Token |
| POST | `/chat/ask_stream` | 统一编排问答（SSE，节点级进度直通） | Bearer Token |
| GET | `/health` | 浅健康检查（探针用） | 无 |
| GET | `/health/ready` | 深度就绪检查（探测 MySQL/Redis/Milvus） | 无 |

### 多模态合规审查（P1，来自 DocAudit）

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/review/upload` | 上传待审查文档并启动多模态解析+审查流水线 | Bearer Token |
| GET | `/review/task/{task_id}/status` | 查询审查任务状态 | Bearer Token |
| GET | `/review/task/{task_id}/events` | 拉取任务阶段事件（parsing→extracting→reviewing→done） | Bearer Token |
| GET | `/review/task/{task_id}/clauses` | 查询抽取出的条条款 | Bearer Token |
| GET | `/review/task/{task_id}/reviews` | 查询自动审查结论（含违规规则） | Bearer Token |
| GET | `/review/task/{task_id}/report` | 生成并获取审查报告 Markdown | Bearer Token |
| POST | `/review/reviews/{review_id}/apply` | 人工复核：采纳/驳回/回滚单条审查 | Bearer Token |
| GET | `/review/rules` | 查询内置合规规则 | Bearer Token |

### 结构化问数（P2，NL2SQL + SQLGuard，来自 ChatBI）

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/nl2sql/ask` | 自然语言 → 结构化数据（NL2SQL + SQLGuard + 只读执行） | Bearer Token |
| POST | `/nl2sql/execute` | 直接执行一条 SQL（幂等只读演示，仍受 SQLGuard 保护） | Bearer Token |
| GET | `/nl2sql/schema` | 查看可用的业务表/字段 schema | Bearer Token |
| POST | `/nl2sql/guard_check` | SQLGuard 安全校验演示（阿里白名单/AST/EXPLAIN） | Bearer Token |

### 知识库

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | `/knowledge_base/` | 查询知识库列表 | Bearer Token |
| POST | `/knowledge_base/` | 创建知识库 | Bearer Token |
| DELETE | `/knowledge_base/{kb_id}` | 删除知识库 | Bearer Token |
| GET | `/knowledge_base/{kb_id}/stats` | 知识库统计 | Bearer Token |

### 文档

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/document/upload` | 上传并解析文档 | Bearer Token |
| GET | `/document/list` | 文档列表（按知识库筛选） | Bearer Token |
| GET | `/document/chunks` | 文档块预览（分页） | Bearer Token |
| DELETE | `/document/{doc_id}` | 删除文档（含 Milvus 向量） | Bearer Token |

### 对话历史

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/conversation/save` | 保存一条对话消息（需本人会话） | Bearer Token |
| GET | `/conversation/history` | 查询对话历史（参数：session_id, limit） | Bearer Token |
| DELETE | `/conversation/clear` | 清空会话历史（需本人会话） | Bearer Token |
| GET | `/conversation/list` | 会话列表（仅当前用户的会话，按最近活跃排序） | Bearer Token |
| POST | `/conversation/new` | 创建新会话（绑定当前用户） | Bearer Token |

### RAG 评估

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/evaluation/generate-dataset` | LLM 生成评估数据集（参数：num_seeds, questions_per_seed） | 管理员 |
| GET | `/evaluation/datasets` | 列出所有评估数据集 | 管理员 |
| GET | `/evaluation/datasets/{name}` | 加载指定数据集内容 | 管理员 |
| POST | `/evaluation/run` | 一键运行完整评估（生成数据集 → RAG → RAGAS） | 管理员 |
| POST | `/evaluation/evaluate` | 对已有数据集运行 RAGAS 评估 | 管理员 |
| GET | `/evaluation/experiments/compare` | 对比多次实验指标（参数 files 或 latest） | 管理员 |
| GET | `/evaluation/experiments` | 列出所有历史评估实验 | 管理员 |
| GET | `/evaluation/experiments/{filename}` | 获取指定实验详情 | 管理员 |

---

## 数据存储设计

### MySQL 表

| 表名 | 说明 |
|------|------|
| `users` | 用户信息（用户名、密码哈希、角色、邮箱、头像、状态、创建时间） |
| `knowledge_bases` | 知识库（名称、描述、分块参数、领域标签） |
| `documents` | 文档记录（所属知识库、文件名、解析状态、块数） |
| `finance_faq` | 高频问答对（问题、答案、领域、类型） |
| `compliance_review_task` | 合规审查任务（文档、阶段、摘要、状态） |
| `docaudit_document` | 待审查文档（名称、类型、文件路径） |
| `docaudit_clause` | 抽取出的条条款（编号、标题、内容、页码回验） |
| `docaudit_review` | 逐条审查结论（违规规则、判罚、HITL 应用状态） |
| `compliance_rule` | 内置合规规则种子（启动时幂等加载 8 条） |

> **结构化问数（P2）**：`config.ini [nl2sql]` 指定两个 MySQL 库——业务库 `biz_demo`（仅由只读账号 `chatbi_ro` 访问，表如 `fact_orders`、`dim_store` 等）与元数据库 `chatbi_meta`（`table_registry` / `column_registry` / `metric_dict` 口径）。SQLGuard 强制 `SELECT` + `LIMIT` + `READ ONLY` + 超时，写操作一律拒绝。

**users 表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT AUTO_INCREMENT | 主键 |
| `username` | VARCHAR(50) UNIQUE | 用户名 |
| `password_hash` | VARCHAR(255) | bcrypt 密码哈希 |
| `role` | ENUM('admin', 'user') | 角色，默认 user |
| `email` | VARCHAR(100) | 邮箱，默认空 |
| `avatar` | VARCHAR(255) | 头像，默认空 |
| `status` | ENUM('active', 'disabled') | 状态，默认 active |
| `created_at` | TIMESTAMP | 创建时间 |
| `updated_at` | TIMESTAMP | 更新时间（自动更新） |

### Redis Key

| Key 格式 | 说明 |
|----------|------|
| `finrag:qa:{md5}` | 问答缓存（TTL 1 天，文档/知识库变更时整体作废） |
| `finrag:session:{sid}` | 多轮对话历史（TTL 7 天，保留最近 50 条，value 含 user_id 字段） |
| `finrag:captcha:{captcha_id}` | 图形验证码（TTL 5 分钟） |
| `finrag:rl:{维度}:{分片}` | 分布式限流计数（认证/问答/LLM 配额） |

### Milvus Collection

固定 schema 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 | 主键（自增） |
| `dense_vector` | FLOAT_VECTOR (1024) | BGE-M3 稠密向量（IVF_FLAT, IP） |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | BGE-M3 lexical 稀疏向量（SPARSE_INVERTED_INDEX, IP） |
| `text` | VARCHAR | 子块原文（向量化时使用的文本，Reranker 输入） |
| `category` | VARCHAR | 领域标签（检索过滤字段） |
| `type` | VARCHAR | 类型标签 |
| `question` | VARCHAR | FAQ 问题原文；文档块存 chunk id（如 `doc_1_parent_0_child_0`） |
| `answer` | VARCHAR | FAQ 答案原文；文档块存父块全文（上下文） |

扩展字段（schema 开启 dynamic field，随行写入）：

| 字段 | 说明 |
|------|------|
| `kb_id` | 所属知识库（文档块行必有，FAQ 行无） |
| `doc_id` | 所属文档记录 id（按文档精确删除向量） |
| `source_file` | 源文件名（旧数据按文件名删除的回退匹配字段） |
| `parent_id` | 父块 id |
| `parent_text` / `metadata` | 父块文本 / 扩展元数据 |

---

## 前端界面

前端为单页应用（`static/index.html`），采用 ChatGPT 风格设计。

### 整体布局

- **左侧深色侧边栏**：Logo、新建对话按钮、历史会话列表、领域选择、导航菜单、用户信息
- **右侧浅色主区域**：对话消息区 + 底部输入框

### 核心功能

| 功能 | 说明 |
|------|------|
| 登录/注册 | Tab 切换，图形验证码，表单校验，紫蓝渐变背景 + 浮动动画 |
| 会话策略 | Token 仅存内存，刷新/关闭页面后重新登录（安全优先） |
| 历史会话 | 侧边栏列表，仅显示当前用户的会话，以首次提问为标题，显示最近活跃时间，支持删除 |
| 消息气泡 | 用户消息靠右（蓝色渐变 + 首字母头像），助手消息靠左（机器人头像 + 白色卡片） |
| 加载动画 | 答案生成前显示跳动圆点 + "答案生成中，请稍候..." |
| 来源展示 | 答案生成完毕后显示"查看来源"按钮，展开引用文档 |
| 消息操作 | 用户消息：复制、删除、显示时间；助手消息：复制、重新生成、点赞、踩 |
| 重新生成 | 替换原回答，不重复保存用户问题 |
| 时间戳 | 每条消息显示发送时间，历史消息保留原始时间 |
| 用户信息 | 底部显示头像（管理员紫色 / 普通用户橙色）、用户名、角色徽章 |
| 管理面板 | 仅管理员可见：统计卡片 + 用户管理表格（搜索、分页、编辑、删除、查看用户会话） |
| 用户会话查看 | 管理员可点击用户查看其所有会话列表，再点击会话查看完整对话历史 |
| 知识库管理 | 卡片式展示，支持创建、删除、文档上传、分块预览 |
| 响应式 | 平板/手机自适应，侧边栏可滑出收起 |
| 领域选择 | 侧边栏标签 + 输入框底部快速选择（自动/银行业务/公司金融/财务会计/通用） |

### 消息交互流程

```
用户输入 → 创建用户消息气泡 → 创建助手气泡（加载动画）
    → SSE 流式接收 token（逐字显示答案）
    → done 事件 → 替换完整答案 → 显示来源按钮
    → 后端自动保存用户消息 + 助手消息到 Redis
```

---

## 测试

```bash
# 运行全部测试
cd D:\FinRag
.venv\Scripts\python.exe -m pytest tests/ -v

# 运行单文件测试
.venv\Scripts\python.exe -m pytest tests/test_bm25.py -v
```

当前测试覆盖：

| 测试文件 | 覆盖内容 |
|----------|---------|
| `test_config.py` | 配置加载与数据类 |
| `test_bm25.py` | BM25 索引构建与检索 |
| `test_embedding.py` | BGE-M3 向量生成 |
| `test_reranker.py` | BGE-Reranker 排序 |
| `test_llm.py` | LLM 调用与流式输出 |
| `test_intent.py` | BERT 意图分类 |
| `test_query_classifier.py` | 分类器完整流程 |
| `test_mysql_db.py` | MySQL 数据访问（Mock） |
| `test_redis_db.py` | Redis 缓存操作（Mock） |
| `test_rag_system.py` | 混合检索流程（Mock） |
| `test_strategy_selector.py` | Agent 策略选择（Mock） |
| `test_bert_train.py` | BERT 微调训练 |
| `test_models.py` | Pydantic 数据模型 |
| `test_document_loaders.py` | 文档加载器 |
| `test_document_processor.py` | 文档解析与分块 |
| `test_text_spliter.py` | 文本切分器 |
| `test_orchestrator.py` | 统一编排层（路由规则/门禁/AnswerGraph 分支） |

> **路由准确率门禁**：`scripts/eval_routing.py` 在 218 题带 Skill 标注的数据集上断言 ≥90%（当前 100%），失败时非零退出码，可纳入 CI。
> **真实服务冒烟**：`POST /api/v1/chat/ask`、`/nl2sql/ask` 等可在启动服务后结合真实 LLM 进行端到端验收（见上文 API 章节）。

---

## 配置说明

配置文件位于项目根目录 `config.ini`，主要配置项：

```ini
[mysql]
host = localhost
port = 3306
user = root
password = root
database = finrag_qa

[redis]
host = localhost
port = 6379
password = 1234
db = 0

[milvus]
host = localhost
port = 19530
database_name = finrag
collection_name = finrag_faq

[llm]
model = qwen3.7-flash
# ⚠️ API Key 不要写在这里，请通过环境变量 DASHSCOPE_API_KEY（或 .env）注入
dashscope_api_key =
temperature = 0.7
# 备用模型（主模型失败时自动降级重试一次，留空不启用）
fallback_model =

[retrieval]
parent_chunk_size = 1200
child_chunk_size = 300
chunk_overlap = 50
retrieval_k = 5
bm25_hit_threshold = 0.75
# 多策略检索（规则预筛路由，关闭后全部直接检索）
multi_strategy_enabled = true
# 规则未命中时对长查询启用 LLM 策略选择器兜底
strategy_llm_fallback = false

[models]
bert_intent_model_path = ./rag_qa/models/bert_intent
bert_base_model_path = ./bert-base-chinese
```

融合新增配置段（节选，完整见 `config.ini.example`）：

```ini
# ---------- 多模态合规审查（融合 DocAudit，P1） ----------
[multimodal]
digital_min_chars = 50        # 页文本层字符数阈值，低于走 OCR
ocr_confidence_route = 0.85   # OCR 平均置信度低于此 → 视觉模型兜底
page_concurrency = 2
upload_max_mb = 50

[compliance]
coverage_threshold = 0.80     # 条款内容页码回验通过率
extract_max_retries = 2
retrieve_top_k = 3
batch_size = 4
dual_temp = true

# ---------- 结构化问数（融合 ChatBI，P2） ----------
[nl2sql]
biz_database = biz_demo
meta_database = chatbi_meta
ro_user = chatbi_ro           # 业务库只读账号（仅 SELECT，见 sql/grant_ro.sql）
ro_password = <本地自填，勿提交>   # 真实密码只写在 config.ini（已被 .gitignore 排除）
max_rows = 1000               # SQLGuard 强制 LIMIT 上限
timeout_ms = 5000             # 查询超时
explain_scan_threshold = 2000000  # EXPLAIN 预检扫描行数阈值
repair_max_rounds = 2         # SQL 生成失败自修复轮数
schema_top_k = 4
embedding_model = qwen3.7-text-embedding-flash
embedding_dim = 1024
use_embedding_recall = true

# ---------- 统一编排（融合 ChatBI/DocAudit，P3） ----------
[orchestrator]
enable_answer_graph = true    # /chat/ask 走 LangGraph AnswerGraph；false=回退老链路
```

环境变量（可选，优先级高于 config.ini）：

- `DASHSCOPE_API_KEY` / `API_KEY` — 覆盖 LLM & Embedding API Key（**推荐方式**；NL2SQL 亦复用同一 Key）
- `NL2SQL_EMBEDDING_MODEL` — 覆盖 LLM 向量模型
- `ORCH_ENABLE_ANSWER_GRAPH` — 覆盖统一编排开关
- `MYSQL_*` / `REDIS_*` / `MILVUS_*` — 覆盖数据库与向量库连接
- `CORS_ORIGINS` — 跨域来源（逗号分隔）
- `LOG_LEVEL` — 日志级别
- 完整列表见 `.env.example`；配置模板见 `config.ini.example`（`config.ini` 已被 `.gitignore` 忽略，请勿提交）

---

## 目录结构

```
FinRag/
├── main.py                  # FastAPI 应用入口
├── config.py                # 配置管理（从 config.ini 读取）
├── config.ini               # 配置文件
├── .jwt_secret              # JWT 签名密钥（持久化，重启后 Token 仍有效）
├── pyproject.toml           # 项目依赖与 pytest 配置
├── init_data.py             # 内置数据初始化脚本
├── create_users_table.py    # users 表初始化（含 role 等字段 + 默认管理员）
├── log_config.py            # 日志配置（控制台 + 文件）
├── models.py                # Pydantic 数据模型（含 UserInfo、UserUpdateRequest）
├── Dockerfile               # Docker 镜像构建
├── docker-compose.yml       # 多容器编排
├── .env.example             # 环境变量模板
├── init_sql/
│   └── init.sql             # 数据库建表 DDL（含 users 表完整字段）
├── db/
│   ├── mysql.py             # MySQL 连接与操作（含用户 CRUD）
│   ├── redis.py             # Redis 连接与操作
│   ├── milvus.py            # Milvus 连接与操作
│   ├── document.py          # 文档记录 CRUD
│   └── chunk.py             # Milvus 向量块操作
├── rag_qa/
│   ├── core/
│   │   ├── query_classifier.py  # BERT 意图分类器
│   │   └── strategy_selector.py # Agent 策略选择
│   ├── edu_document_loaders/    # 文档加载器
│   ├── edu_text_spliter/        # 文本分块器
│   ├── multimodal/              # 多模态分级解析（融合 DocAudit, P1）
│   ├── review/                  # 合规审查流水线 + LangGraph 审查（P1）
│   ├── nl2sql/                  # NL2SQL + SQLGuard（融合 ChatBI, P2）
│   │   ├── sqlguard.py          # 三层纵深防御（AST/EXPLAIN/只读）
│   │   ├── schema_retriever.py  # Schema 关键词+向量召回
│   │   ├── generation.py        # SQL 生成 + 自修复重写
│   │   ├── executor.py          # 只读执行（READ ONLY + LIMIT + 超时）
│   │   └── meta_store.py        # 元数据访问层
│   ├── orchestrator/            # 统一编排 LangGraph AnswerGraph（P3）
│   │   ├── graph.py             # StateGraph + 条件边
│   │   ├── route.py             # 意图→Skill 路由
│   │   ├── nodes.py             # rag/nl2sql/chitchat 节点
│   │   └── state.py             # AnswerState 统一状态
│   └── models/
│       └── bert_intent/         # 微调 BERT 模型
├── routers/
│   ├── auth.py              # 认证接口（注册/登录/验证码/用户信息）
│   ├── chat.py              # 问答接口（SSE 流式 + 自动保存对话）
│   ├── admin.py             # 管理员接口（用户管理/系统统计/用户会话查看）
│   ├── knowledge_base.py    # 知识库管理接口
│   ├── document.py          # 文档上传/解析/预览接口
│   ├── conversation.py      # 多轮对话接口（含会话隔离）
│   ├── evaluation.py        # RAG 评估接口（RAGAS 数据集生成/评估/结果查看）
│   ├── review.py            # 多模态合规审查接口（融合 DocAudit, P1）
│   └── nl2sql.py            # 结构化问数接口（NL2SQL + SQLGuard, P2）
├── services/
│   ├── bm25.py              # BM25 检索服务
│   ├── embedding.py         # BGE-M3 向量化服务
│   ├── reranker.py          # BGE-Reranker 重排序服务
│   ├── llm.py               # LLM 调用服务
│   ├── intent.py            # 意图识别服务
│   └── evaluation.py        # RAG 评估服务（RAGAS 数据集生成/RAG 评估/结果存储）
├── evals/                    # RAG 评估工作目录
│   ├── datasets/             # LLM 生成的评估数据集
│   └── experiments/          # 评估实验结果
├── scripts/
│   ├── train_intent_bert.py # BERT 微调训练脚本
│   ├── eval_routing.py      # 路由准确率评测门禁（默认 90%，P3）
│   └── route_skill_eval.json # 218 题带 Skill 标注的评测数据集（P3）
├── static/
│   └── index.html           # 前端单页应用（ChatGPT 风格）
├── tests/
│   ├── conftest.py          # pytest 公共 fixtures
│   └── test_*.py            # 单元测试文件
├── data/                    # 内置数据（FAQ、知识文档）
├── logs/                    # 日志输出目录
├── bge-m3/                  # BGE-M3 本地模型
├── bge-reranker-v2-m3/      # BGE-Reranker 本地模型
└── bert-base-chinese/       # BERT 基础模型
```
