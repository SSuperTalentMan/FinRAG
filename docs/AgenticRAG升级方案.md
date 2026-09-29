# FinRag 升级 Agentic RAG 方案（MCP 工具层 + Agent 循环编排）

生成时间：2026-09-12。回答的问题：能否把 FinRag 的 RAG 封装成 MCP 工具，由 Agent 调用，把经典 RAG 升级为 Agentic RAG。

## 一、结论

可行，且 FinRag 已具备约 80% 前提——三类能力（RAG 问答 / NL2SQL 问数 / 合同审查）本就是服务化接口。本轮补齐两层：

1. **路径 A：MCP 工具层**（`rag_qa/mcp_layer/`）—— FinRag 变成标准 MCP Server，任何支持 MCP 的 Agent（Claude / WorkBuddy / 自建）都能调用；
2. **路径 B：Agentic 编排**（`rag_qa/orchestrator/agentic.py`）—— AnswerGraph 的 rag/nl2sql 分支可切换为 Agent 循环（多跳检索 + 跨工具组合），配置开关默认关闭。

## 二、路径 A：MCP 工具层

### 交付物

| 文件 | 职责 |
|---|---|
| `rag_qa/mcp_layer/registry.py` | 管控层：未知工具拒绝 → 身份越权拒绝 → 参数校验 → 线程池超时隔离 → 审计脱敏（字段名+值模式双重） |
| `rag_qa/mcp_layer/tools.py` | 6 个工具，HTTP 薄壳调常驻 FastAPI 服务（纯标准库 urllib） |
| `rag_qa/mcp_layer/server_stdio.py` | 最小 MCP stdio Server（换行分隔 JSON-RPC 2.0，只实现 initialize / tools/list / tools/call） |
| `rag_qa/mcp_layer/client.py` | 子进程客户端（握手 + 调用，跳过脏输出行） |
| `routers/chat.py` 新增 `POST /api/v1/chat/search` | 只检索不生成的端点，供 `finrag.kb_search` 使用 |
| `scripts/mcp_demo.py` | 端到端演示：管控负例 + 协议往返 + 降级路径 |

### 工具清单（6 个）

| 工具 | 后端 | 说明 |
|---|---|---|
| `finrag.ask` | `POST /api/v1/chat/ask` | 端到端统一问答（自动路由），返回带溯源答案 |
| `finrag.kb_search` | `POST /api/v1/chat/search`（新增） | 只检索不生成，返回带来源与相关分的块列表，供 Agent 自评是否够用 |
| `finrag.sql_query` | `POST /api/v1/nl2sql/ask` | 自然语言问数（NL2SQL + SQLGuard + 只读执行） |
| `finrag.review_upload` | `POST /api/v1/review/upload` | 发起审查长任务，立即返回 task_id |
| `finrag.review_status` | `GET /api/v1/review/task/{id}/status` | 轮询审查进度 |
| `finrag.review_report` | `GET /api/v1/review/task/{id}/report` | 获取完整审查报告 |

### 关键设计决策

1. **MCP server 是薄壳，不加载模型、不连库**。BGE-M3 冷加载约 1 分钟，若 MCP server 自己加载，首次调用必超时；改为 HTTP 转发常驻服务，工具超时可设 30~120s。前提：FinRag 服务必须在运行。
2. **stdout 改道先于一切 import**（`server_stdio.py` 顶部）：MCP stdio 把 stdout 独占为协议通道，任何库的一行日志输出都会让客户端解析失败。
3. **长任务异步化**：审查是分钟级任务，`review_upload` 只返回 task_id，由 Agent 轮询 `review_status`，不阻塞决策循环。
4. **鉴权复用现有 JWT**：环境变量 `FINRAG_TOKEN` 注入 Bearer Token；`FINRAG_BASE_URL` 指定服务地址（默认 `http://127.0.0.1:8000`）；`FINRAG_CALLER` 标识调用方身份（默认 agent，注册表按 scopes 校验）。
5. **审计落 JSONL**（`logs/mcp_audit.jsonl`，best-effort）：记录 tool/caller/脱敏参数/结果/耗时；密钥类值按模式正则打码。
6. **绝不 mock**：服务不可达时返回明确错误（含排查提示），不用假数据充数。

### 接入外部 Agent 的配置示例

```json
{
  "mcpServers": {
    "finrag": {
      "command": "D:/FinRag/.venv/Scripts/python.exe",
      "args": ["-m", "rag_qa.mcp_layer.server_stdio"],
      "env": {
        "PYTHONPATH": "D:/FinRag",
        "FINRAG_BASE_URL": "http://127.0.0.1:8000",
        "FINRAG_TOKEN": "<JWT，可用 routers.auth._create_token 签发>",
        "FINRAG_CALLER": "claude-desktop"
      }
    }
  }
}
```

## 三、路径 B：Agentic 编排（Agent 循环）

### 与固定编排的区别

固定路由（现状）：route 一次定型，rag 与 nl2sql 互斥，检索一轮定胜负。
Agent 循环（新增）：LLM 看到工具描述（kb_search / sql_query / finish）+ 每跳观察结果，自主决定下一跳——最多 `agentic_max_hops`（默认 3）跳后必须收尾。

真实收益三处：
1. **多跳检索**：第一轮召回偏题时自主改写重查（观察命中数与最高分后决策）；
2. **跨工具组合**：「存款保险最高赔多少，再查下今年赔付笔数」可串 RAG + 问数，最终 LLM 综合两部分资料作答；
3. **结果自评**：Agent 判断资料是否足够，不足继续检索，足够立即 finish（不空转）。

### 安全设计（金融场景确定性优先）

- **默认关闭**：`config.ini [orchestrator] agentic_mode = false`（或环境变量 `ORCH_AGENTIC_MODE=1`）。关闭时图结构与行为与升级前完全一致（已验证开关分流）。
- **失败必回退**：LLM 决策失败 / 跳数用尽未 finish / 工具链路异常 / 未收集到任何资料——一律回退固定路由节点（rag 或 nl2sql，按 route 已判定结果），绝不把「Agent 失败」当答案返回。
- **工具与固定路由同源**：agentic 节点内调用的是与 `rag_retrieve_node` / `nl2sql_node` 完全相同的 `_build_context` / `Nl2SqlService`，行为一致性有保证；鉴权/审计/配额沿用现有体系。
- **上下文控制**：每跳观察回喂截断 400 字，跳数上限 3，防止上下文膨胀。

### 相关文件

- `rag_qa/orchestrator/agentic.py`（新增）：决策 prompt、工具执行、回退逻辑、收尾作答
- `rag_qa/orchestrator/graph.py`：注册 agentic 节点 + 条件边按 agentic_mode 分流
- `config.py` / `config.ini`：`agentic_mode` / `agentic_max_hops` 配置

## 四、验证证据（2026-09-12，离线可复跑）

`scripts/mcp_demo.py` 全绿：

- **A 管控负例 5/5 拦截**：未知工具 / 缺必填参数 / bool 冒充 integer / 身份越权 / 超时隔离（1s 超时不等 5s 任务）；审计脱敏验证 `api_key`/`phone` 均打码
- **B 协议往返**：子进程握手成功（finrag-mcp / protocol 2024-11-05），`tools/list` 返回 6 个工具，未知工具调用返回 isError
- **C 降级路径**：服务未启动时 `finrag.kb_search` 返回明确错误（含排查提示），无 mock 数据
- **Agentic 图结构**：`build_answer_graph()` 10 节点含 agentic；开关分流断言全过（关=原路由，开=rag/nl2sql 走 agentic，chitchat/multimodal/review 不变）
- 全部 10 个改动文件 `py_compile` 通过

## 五、待办（需要运行环境）

1. **服务端真实验证**：启动服务后重跑 `scripts/mcp_demo.py` 的 C 部分（需先设 `FINRAG_TOKEN`）；再开 `agentic_mode = true` 跑跨工具组合问题（「存款保险最高赔多少，再查下今年赔付笔数」）验证多跳与综合作答，观察 `meta.agent_hops` 轨迹。
2. **端到端评估**：agentic 模式下跑 `scripts/eval_routing.py` / `eval_nl2sql.py` 对比固定路由指标，确认无回归后再考虑默认开启。
3. **成本提醒**：agentic 模式每问多 1~3 次 LLM 决策调用（延迟 +1~3s），配额扣减在入口已有，无需改造。

## 六、数据边界与风险红线

- MCP 工具层只转发**本系统真实拥有**的能力（自建知识库 / biz_demo 业务库 / DocAudit 审查流水线），不引入任何外部数据源，无 mock。
- 审查结论、问数结果均来自真实流水线；Agent 循环失败即回退固定路由，不存在「编一个答案」的路径。
- FINRAG_TOKEN 仅经环境变量注入，不落盘、不进日志（审计层有值模式脱敏兜底）。
