# 融合功能手工测试用例（ChatBI 问数 + DocAudit 审查 + 统一编排）

- 适用版本：2026-09-08（含 P1 口径治理、`/chat/stream` 500 修复）
- 基址：`http://localhost:8000/api/v1`
- 认证：所有接口需 `Authorization: Bearer <token>`，token 获取见第 0 节

## 0. 准备：签一个测试 token

```powershell
# 在项目根目录执行（复用 .jwt_secret 签 token，绕过图形验证码）
.venv\Scripts\python.exe -c "from routers.auth import _create_token; print(_create_token(1, 'admin', 'admin'))"
# 普通用户角色（测权限用）：
.venv\Scripts\python.exe -c "from routers.auth import _create_token; print(_create_token(2, 'tester', 'user'))"
```

把输出的 token 存到变量（PowerShell）：

```powershell
$T = "<粘贴token>"
$H = @{ Authorization = "Bearer $T" }
$BASE = "http://localhost:8000/api/v1"
```

---

## A. 统一编排 `/chat/ask`（路由正确性）

请求体统一为 `{"message": "...", "session_id": ""}`，响应里看 `skill` 字段。

| #  | 输入                          | 期望                                                                            |
| -- | --------------------------- | ----------------------------------------------------------------------------- |
| A1 | `上个月各大区的销售额是多少`             | skill=**nl2sql**，status=ok，返回 SQL+数据行                                         |
| A2 | `存款保险最高赔付多少`                | skill=**rag**（历史误判 insurance，已加 DOMAIN_OVERRIDE，不得进 nl2sql）                   |
| A3 | `股票发行注册制改革的背景是什么`           | skill=**rag**，且**不得** BM25 早退（sources 应含文档 chunk，confidence 不等于 BM25 softmax） |
| A4 | `你好，你是谁？`                   | skill=**chitchat**                                                            |
| A5 | `图里写了什么`                    | skill=**multimodal**，返回上传引导（不并入 rag）                                          |
| A6 | `帮我审查这份采购合同`                | 走**对话内审查**受理（创建 review 任务或引导上传）                                               |
| A7 | `客单价 Top10 门店`（经 /chat/ask） | 同 A1 且口径正确（对照 B2）                                                             |

```powershell
foreach ($q in @('上个月各大区的销售额是多少','存款保险最高赔付多少','你好，你是谁？')) {
  $r = Invoke-RestMethod -Uri "$BASE/chat/ask" -Method Post -Headers $H `
        -ContentType 'application/json; charset=utf-8' -Body (@{message=$q; session_id=''} | ConvertTo-Json)
  "$q => skill=$($r.data.skill) status=$($r.data.status)"
}
```

## B. 问数 NL2SQL（P1 口径治理验证重点）

### B1. 派生指标口径（few-shot + caliber_check 生效性）

| #  | 问题             | 期望（看响应 data.sql）                                                                                         |
| -- | -------------- | -------------------------------------------------------------------------------------------------------- |
| B1 | `客单价最高的前10个门店` | SQL 含 `SUM(pay_amount)` + `COUNT(DISTINCT ... order_id)`；**不得出现 `AVG(pay_amount)`**（命中即说明 post-check 失效） |
| B2 | `退货率最高的五个商品`   | SQL **必须** JOIN `fact_refunds`，按订单数占比（非 `SUM(refund_amount)` 金额比）；秒级返回（预聚合写法，不得 3024 超时）                 |
| B3 | `本季度的复购率`      | SQL 含 `QUARTER(...)` 且含 `YEAR(...)`                                                                      |
| B4 | `本月 GMV 同比增长`  | 含去年对比（`INTERVAL 1 YEAR` 或 `YEAR(...)-1`）                                                                 |
| B5 | `上个月销售额环比`     | 两个相邻月份对比                                                                                                 |

### B2. 正确拒答（biz_demo 无表，禁编造）

| #  | 问题                                        | 期望                                               |
| -- | ----------------------------------------- | ------------------------------------------------ |
| B6 | `各门店上月业绩完成率`                              | status=**refused**（caliber_check 第 5 类强制 REFUSE） |
| B7 | `毛利率/毛利额是多少`、`库存周转率`                      | refused，理由说明无对应表                                 |
| B8 | `帮我删除所有订单` / `UPDATE fact_orders SET ...` | `/nl2sql/guard_check` 返回拒绝（只读防线）                 |

```powershell
$r = Invoke-RestMethod -Uri "$BASE/nl2sql/ask" -Method Post -Headers $H `
      -ContentType 'application/json; charset=utf-8' -Body (@{question='客单价最高的前10个门店'} | ConvertTo-Json)
$r.data.sql; $r.data.status; $r.data.rows | Select-Object -First 3

# 写操作防线（应被拒）
Invoke-RestMethod -Uri "$BASE/nl2sql/guard_check" -Method Post -Headers $H `
  -ContentType 'application/json; charset=utf-8' -Body (@{sql='DELETE FROM fact_orders'} | ConvertTo-Json)
```

### B3. 安全/权限

| #   | 用例                             | 期望                      |
| --- | ------------------------------ | ----------------------- |
| B9  | 普通用户 token 查 `/nl2sql/schema`  | 只返回其角色白名单内的表            |
| B10 | SQL 含 `SLEEP(5)` / 多语句 `;DROP` | guard 拒绝，且执行耗时正常（无延时注入） |

## C. 合规审查 DocAudit（多模态解析 + 风险分级）

准备一份含典型风险条款的 PDF（如含「甲方可随时单方解除合同」「违约须支付合同总额 30% 罚金」「乙方放弃一切索赔权」）。

```powershell
$up = Invoke-RestMethod -Uri "$BASE/review/upload" -Method Post -Headers $H `
       -Form file@(Get-Item 'D:\测试合同.pdf')
$tid = $up.data.task_id
# 轮询（递增 start 拉增量事件，final=true 结束）
Invoke-RestMethod -Uri "$BASE/review/task/$tid/events?start=0" -Headers $H
```

| #  | 用例              | 期望                                                                                            |
| -- | --------------- | --------------------------------------------------------------------------------------------- |
| C1 | 全流程             | status 流转 parsing→extracting→reviewing→completed；扫描版 PDF 走 OCR/VL 页（events 里可见 parser_source） |
| C2 | 风险分级            | 「随时单方解除」→ **violation/high**；「违约金 30%」→ 至 least medium；「自动续展无退出权」→ violation                  |
| C3 | 拉结果             | `/task/{id}/clauses`、`/task/{id}/reviews`、`/task/{id}/report`（Markdown 报告）                    |
| C4 | HITL            | 低置信条款 hitl_status=pending → approve/reject 后收敛，任务不被永久卡住                                       |
| C5 | **越权（IDOR 回归）** | 用普通用户 B 的 token 查用户 A 的 task_id → **404**（防枚举），不能 200/403 泄露存在性                               |
| C6 | 上传白名单           | `.txt`/`.exe` → 400；>50MB → 413（流式判，不占内存）                                                     |

## D. 老链路回归（今日 500 修复验证）

| #  | 用例                       | 期望                                                                                                    |
| -- | ------------------------ | ----------------------------------------------------------------------------------------------------- |
| D1 | `POST /chat/`（老前端报文）     | 200 + 响应头 `Deprecation: true`、`Warning: 299 - "Deprecated: migrate to /chat/ask ..."`（纯 ASCII，不再 500） |
| D2 | `POST /chat/stream`（SSE） | 正常出流，无 `latin-1` 报错；日志里弃用告警内容完整（非字面 `%s`）                                                             |
| D3 | 会话连续性                    | `/chat/ask` 带 session_id 两轮追问（如先问「华东大区上月单量」再问「那上上个月呢」）→ 上下文生效，`/conversation/history` 落库              |

## E. 服务就绪自检

```
GET /health          → 200
GET /health/ready    → 200 且 redis/mysql/milvus/bge_m3/reranker 全 ok（BERT 可 degraded）
```

## F. 实时行情注入（P5 新增，2026-09-12）

| #  | 用例                            | 期望                                                                                          |
| -- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| F1 | `/chat/ask` 问「现在的股市行情如何」      | `sources[0].type = realtime_market`，来源为真实生效源（腾讯/东财），`question` 带数据时间；答案含真实指数点位与「数据时间」  |
| F2 | `/chat/ask` 问「今天上证指数涨跌多少」       | 答案给出上证指数具体点位与涨跌幅，且**不编造**盘中数据；休市日明确标注「最近一个交易日收盘数据」                                           |
| F3 | 对照：问「存款保险最高赔付多少」              | **行为与升级前一致**，`sources` 无 `realtime_market`，答案 50 万元限额——注入不得污染制度性问题                              |
| F4 | 对照：问「什么是指数基金」「股票发行注册制改革」      | 不触发行情注入（`sources` 无 `realtime_market`），避免易误触发问句被塞行情数据                                          |
| F5 | **缓存旁路**：连问两次同一条行情问题           | 两次答案的数据时间应各自新鲜（不被 24h 缓存冻住）；Redis 中 `finrag:qa:*` / `ask:*` 不应出现该问句的键                        |
| F6 | **BM25 让路**：问行情类问题              | 走检索+行情注入链路，不得被 BM25 FAQ 早退直接返回静态答案                                                           |
| F7 | 知识库零命中但含行情                     | 返回行情上下文并作答，**不得**回答「未在知识库中找到相关资料」                                                            |
| F8 | **断网降级**：临时改 `[market_data] provider=tencent` + 错误 URL | 主链路照常回答（纯知识库路径），日志一条 WARNING，**不报错、不 500**                                                   |

> F5/F6 是本次改动的关键回归点：行情类问句必须缓存旁路、BM25 让路，否则「注入做了但用户永远看不到」。
> 复验脚本：`scripts/test_market_data.py`（离线，识别/取数/缓存/降级）、`scripts/test_market_integration.py`（注入接线/回归对照）。
> 方案细节见 `docs/实时行情注入方案_P5.md`。

> 建议**先跑 D1/D2**（验证重启后今天的修复生效），再按 A→B→F→C 顺序覆盖；B 组是 P1 口径治理的验收重点，命中 B1 的 AVG 或 B6 的编造 SQL 即回归失败。
