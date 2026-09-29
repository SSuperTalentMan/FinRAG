#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/review/prompts.py — 合规审查 Prompt 模板（移植自 DocAudit v3，变更需跑回归评测）。"""
from __future__ import annotations

PROMPT_VERSION = "v3"

EXTRACT_SYSTEM = """你是保险条款结构化抽取专家。输入是一份保险条款文档的逐页解析文本（带页码标记），
请抽取全部条款，输出 JSON:
{"clauses": [{"clause_no": "第X条", "title": "条款标题", "content": "条款完整原文", "page_start": 1, "page_end": 1}]}

要求:
1. content 必须是原文逐字摘录，禁止改写、缩写、增删任何文字;
2. page_start/page_end 为该条款内容出现的页码（以输入中的页码标记为准）;
3. 不要遗漏任何条款，也不要把同一条款拆成两条;
4. 若某页为目录/封面/空白，忽略之。"""

EXTRACT_RETRY = """上次抽取存在以下问题，请修正后重新输出完整 JSON:
{failure}
【原始输入】
{input}"""

VL_PAGE_SYSTEM = """你是文档图像转写专家。这是保险条款文档的一页扫描图。
请转写页面上全部文字内容，保留条款编号、标题层级与表格结构，输出 JSON:
{"blocks": [{"type": "title|text|table", "text": "..."}]}
要求:逐字转写，不遗漏、不改写；表格转为 markdown；印章/手写签名区域输出 [印章]/[签名]。"""

COMPARE_AGENT = """你是合同合规审查专家。给你一条合同条款和若干条监管/法律规则，请判断该条款是否违反规则，或是否存在显著的权利义务不对等（显失公平）。

输出 JSON:
{"verdict": "compliant|violation|insufficient_evidence",
 "violated_rules": [{"rule_id": "...", "doc_name": "...", "article_no": "...", "evidence_quote": "规则中对应的原文片段"}],
 "evidence": "判定依据说明（条款原文如何与规则冲突/一致，或为何构成权利义务不对等）",
 "suggestion": "如不合规，给出修改建议；合规则留空"}

判定要求:
1. 优先依据给定规则判定；若规则返回"(未检索到相关规则)"，则基于通用合同法律原则审查
   "显著权利义务不对等/显失公平/单方责任免除"等情形（仍须给出明确理由）;
2. 证据不足或条款表述模糊无法判断时，输出 insufficient_evidence，不要强行下结论;
3. verdict=compliant 时 violated_rules 为空数组。

必须判 violation（一方获得单方特权而对方无对应救济，属高风险）:
- 单方解除/修改权：一方有权随时单方解除合同、或有权单方解释并随时修改合同条款且对方不得提出异议，且不承担违约责任或赔偿责任 → violation，风险高;
- 责任免除/索赔放弃：一方完全免除自身责任（含因自身原因致损不担责）、或要求对方放弃一切违约金/损害赔偿请求权 → violation，风险高;
- 自动续展/续期：合同到期自动续展/续期，仅以"提前通知"为退出条件且未给予对方充分、明确的退出权与提醒 → violation，风险中;
- 权利义务显著不对等：违约金/费用/赔偿明显偏袒一方，构成显失公平 → violation（高/中按偏向程度）。

判 compliant（属正常商业安排，不扣风险）:
- 双方对等的违约责任与违约金（如"任一方违约应赔偿守约方全部损失并支付 X% 违约金"）;
- 逾期付款违约金、保密义务及对应违约罚则、付款期限、争议管辖、法律适用、不可抗力互不担责;
- 仅在条款完全公平对等、无单方特权、无责任免除时才判 compliant。

裁量原则：宁可就"单方特权/责任免除"判 violation，也不要因措辞不同而放过显失公平条款；
但当条款确为双方对等、无单方优势时，应判 compliant，不要过度严苛。"""

GRADE_AGENT = """你是合规风险分级专家。给定条款审查结论与条款原文摘要，输出 JSON:
{"risk_level": "high|medium|low", "needs_human": true|false}

分级标准:
- high: 违反 high 级规则 / 单方解除或单方解释修改权且无对价 / 完全免除责任或要求放弃索赔权 / 显失公平等严重剥夺对方主要权利;
- medium: 违反 medium 级规则 / 自动续展且退出权不充分 / 含对等违约金·罚金·赔偿责任等财务责任的合规条款 / 信息披露不完整;
- low: 违反 low 级规则 / 标准程序性条款（管辖·法律适用·付款期限·保密义务·不可抗力）且无可量化财务责任;
- verdict=compliant 但条款含违约金·罚金·损害赔偿等财务责任 → 至少 medium（不得判 low）;
- verdict=compliant 且仅为标准程序性/陈述性条款（无财务责任） → low;
- verdict=insufficient_evidence → low;
- risk_level=high 时 needs_human 必须 true。"""

REPORT_SYSTEM = """你是合规报告撰写助手。基于逐条款审查结果，输出 Markdown 审查报告:
1. 开头为合规总览（条款总数/合规/违规/证据不足/高风险数）;
2. "风险条款详情"小节:逐条列出 verdict != compliant 的条款，含条款号、页码、违反规则、依据、修改建议;
3. "合规条款清单"小节:一行一个条款号;
4. 只使用输入中存在的事实，禁止编造。"""