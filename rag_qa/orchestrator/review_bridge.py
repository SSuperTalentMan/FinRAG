#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/orchestrator/review_bridge.py — 对话 ↔ 文档审查（DocAudit）的桥接层。

融合前，审查能力只有一个入口：/review/upload 上传文件。用户在对话框里问
"帮我审这份合同"时，编排层无处可去，只能回一句"请去上传页"。这里补上闭环：

    对话里的待审正文 ──(fitz 合成 PDF)──▶ ComplianceService.create_task ──▶ 后台流水线
                                                                            └──▶ task_id

关键取舍
--------
1. **为什么合成 PDF 而不是把文本直接喂给审查图**：DocAudit 的条款抽取有页码回验
   （coverage_threshold，抽取的条款必须能在原文页码上找到出处），没有真实分页会
   导致回验失败。用 fitz 按页写入文本，即可零改造复用整条
   解析 → 抽取 → 审查 → 报告 链路，审查报告同样可溯源到页码。
2. **为什么这里只做"受理"不做"等待"**：审查是分钟级长任务，对话接口不能挂起等待。
   建任务后立刻返回 task_id，进度由用户追问或前端轮询 /review/task/{id}/status。
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# 判定"用户粘了待审正文"的最小连续字数：短于这个几乎只能是提问本身
_INLINE_TEXT_MIN = 60
# 切分指令与正文：中文冒号/换行后的长段落视为正文
_SPLIT_INSTRUCTION = re.compile(r"[\n\r]+|：|:|如下|以下内容|正文[:：]?")


def accepted_inline_text(question: str) -> str:
    """从问题中提取可能被粘贴进来的待审正文，没有则返回空串。

    思路：按换行/冒号切成片段，取最长且超过阈值的片段。这样"帮我审查一下：{正文}"
    与"请看看下面这段有没有风险\n{正文}"都能正确剥离指令前缀。
    """
    q = (question or "").strip()
    if len(q) < _INLINE_TEXT_MIN:
        return ""
    parts = [p.strip() for p in _SPLIT_INSTRUCTION.split(q)]
    # 过滤掉明显是指令的短片段（"帮我审查一下"之类）
    cands = [p for p in parts if len(p) >= _INLINE_TEXT_MIN]
    if not cands:
        return ""
    body = max(cands, key=len)
    # 去掉可能残留的引导词尾巴
    body = re.sub(r"^(请|帮我|麻烦|麻烦你|看看|审一下|审查一下)", "", body).strip()
    return body if len(body) >= _INLINE_TEXT_MIN else ""


def build_review_guide() -> str:
    """未携带正文时的引导话术（明确能力边界与操作路径，不编造结论）。"""
    return (
        "我可以做文档合规审查：上传 PDF 或图片后，系统会完成版面解析"
        "（数字文本 / OCR / 视觉兜底三级）、条款结构化抽取、逐条合规判定与风险分级，"
        "并生成可下载的审查报告；低置信结论会转人工复核。\n\n"
        "两种方式可以开始：\n"
        "1. 在「合规审查」页上传文件；\n"
        "2. 直接把合同条款粘贴到对话框，我会立即建任务并在后台审查（建议一次不超过 8000 字）。"
    )


def render_text_pdf(text: str, title: str = "对话内提交的待审文本") -> bytes:
    """把纯文本排成分页 PDF（保留中文），供 DocAudit 解析链路消费。

    字体用 PyMuPDF 内置简中字体 "china-s"（Droid Sans Fallback），无需外挂字体文件，
    离线可用。放不下的部分自动续页。
    """
    import fitz  # PyMuPDF —— 延迟导入，只在真正受理审查时才加载

    font = "china-s"
    doc = fitz.open()
    try:
        remaining = text
        page_no = 0
        while remaining:
            page = doc.new_page()
            page_no += 1
            rect = fitz.Rect(56, 64, 539, 750)  # A4 左右留白
            if page_no == 1 and title:
                page.insert_textbox(
                    fitz.Rect(56, 48, 539, 74), title, fontname=font, fontsize=12,
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                rect = fitz.Rect(56, 84, 539, 750)
            used = page.insert_textbox(
                rect, remaining, fontname=font, fontsize=10.5, lineheight=1.4,
                align=fitz.TEXT_ALIGN_LEFT,
            )
            if used <= 0:
                # 极端情况（字体缺失等）兜底：整页塞不进去就直接截断，避免死循环
                page.insert_textbox(rect, remaining[:1500], fontname=font, fontsize=10.5)
                break
            # insert_textbox 返回未排下部分的字符数（float），切片必须转 int
            remaining = remaining[int(used):].lstrip("\n")
            if page_no > 200:  # 硬上限，防御异常输入
                break
        return doc.tobytes(deflate=True)
    finally:
        doc.close()


def _inline_doc_name(text: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    return f"对话内文本_{stamp}_{short}.pdf"


async def submit_inline_review(text: str, user: str = "") -> tuple[int, str]:
    """受理对话内提交的待审文本，返回 (task_id, doc_name)。

    复用 routers/review.py 中的审查服务单例（规则检索器只加载一次），
    任务归属写当前用户名，保证后续按属主校验能看到自己的报告。
    """
    from rag_qa.review.compliance_service import spawn_pipeline
    from routers.review import _get_service

    pdf_bytes = render_text_pdf(text)
    doc_name = _inline_doc_name(text)
    svc = _get_service()
    task_id = await svc.create_task(pdf_bytes, doc_name, "pdf", user or "chat")
    spawn_pipeline(svc, task_id)
    logger.info("对话内审查任务已创建 task=%s user=%s chars=%s", task_id, user, len(text))
    return task_id, doc_name
