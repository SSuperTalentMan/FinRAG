#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_rag_robustness.py — 第二批优化（韧性与生成质量）单元测试
覆盖：输入净化、system/user 分层防注入、Context token 预算截断、
LLM 故障降级提取、对话历史滑窗。
"""
import json
from unittest.mock import patch, MagicMock

from rag_qa.core.prompts import RAGPrompts, sanitize_input
from routers.chat import (
    _estimate_tokens, _truncate_context, _extract_fallback_from_context,
    _load_recent_history,
)


class TestSanitizeInput:
    """输入净化（#9）：移除控制字符、截断超长、去除首尾空白。"""

    def test_removes_control_chars(self):
        assert sanitize_input("abc\x00\x1bdef") == "abcdef"

    def test_strips_whitespace(self):
        assert sanitize_input("  你好  ") == "你好"

    def test_keeps_newline_and_tab(self):
        assert sanitize_input("a\nb\tc") == "a\nb\tc"

    def test_truncates_long_input(self):
        out = sanitize_input("x" * 3000)
        assert len(out) <= 2001  # 2000 + 省略号

    def test_empty_input(self):
        assert sanitize_input("") == ""
        assert sanitize_input(None) == ""


class TestAnswerMessagesLayering:
    """Prompt 统一管理（#5/#9）：system/user 物理隔离 + 注入防御声明。"""

    def test_system_user_separated(self):
        msgs = RAGPrompts.answer_messages("问题", "背景", "banking")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # 用户输入被标签包裹，且与系统指令分离
        assert "<user_input>" in msgs[1]["content"]
        assert "问题" in msgs[1]["content"]
        # 注入防御声明与用户输入同层（user），说明标签内是数据而非指令
        assert "不可执行" in msgs[1]["content"]

    def test_empty_context_honest(self):
        msgs = RAGPrompts.answer_messages("问题", "", "banking")
        assert "无相关背景信息" in msgs[0]["content"]

    def test_history_included(self):
        msgs = RAGPrompts.answer_messages("问题", "背景", "banking", "用户: 你好\n助手: 你好")
        assert "对话历史" in msgs[0]["content"]


class TestGenerateAnswerLayering:
    """generate_answer 应把分层 messages 传给 LLM，且净化用户输入。"""

    @patch("services.llm.chat_completion")
    def test_passes_layered_messages(self, mock_chat):
        from services.llm import generate_answer
        mock_chat.return_value = "回答"
        generate_answer("问题", "背景", "banking", "历史")
        msgs = mock_chat.call_args[0][0]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "<user_input>" in msgs[1]["content"]

    @patch("services.llm.chat_completion")
    def test_sanitizes_question(self, mock_chat):
        from services.llm import generate_answer
        mock_chat.return_value = "回答"
        generate_answer("abc\x00def", "背景", "banking")
        msgs = mock_chat.call_args[0][0]
        assert "\x00" not in msgs[1]["content"]


class TestContextBudget:
    """Context token 预算（#6）：估算 + 段落边界截断。"""

    def test_estimate_tokens(self):
        assert _estimate_tokens("ab") >= 1
        # 中文约 2.5 字/token：10 字 ≈ 4 token
        assert _estimate_tokens("一二三四五六七八九十") == 4

    def test_no_truncation_within_budget(self):
        short = "你好" * 10
        assert _truncate_context(short) == short

    def test_truncates_over_budget(self):
        long_context = "\n\n".join("段落" * 200 for _ in range(200))
        out = _truncate_context(long_context, budget=100)
        assert _estimate_tokens(out) <= 110  # 允许边界段误差
        assert len(out) < len(long_context)


class TestLLMDegradation:
    """LLM 故障降级（#1）：从检索 context 提取首条回答作兜底。"""

    def test_extract_fallback_from_context(self):
        ctx = "【问题】Q1（来源：x）\n【回答】这是第一条回答\n\n【问题】Q2\n【回答】第二条"
        assert _extract_fallback_from_context(ctx) == "这是第一条回答"

    def test_fallback_empty_context(self):
        assert "不可用" in _extract_fallback_from_context("")


class TestHistoryWindow:
    """对话历史滑窗（#7）：仅取最近 N 轮，按 token 预算截断。"""

    @patch("routers.chat.get_redis")
    def test_load_recent_history(self, mock_redis):
        fake = MagicMock()
        # 10 条消息（5 轮），滑窗保留最近 8 条（4 轮）
        msgs = []
        for i in range(1, 6):
            msgs.append({"role": "user", "content": f"问题{i}", "domain": "x", "ts": i * 2 - 1})
            msgs.append({"role": "assistant", "content": f"回答{i}", "domain": "x", "ts": i * 2})
        fake.get.return_value = json.dumps({"messages": msgs})
        mock_redis.return_value = fake
        out = _load_recent_history("sess1", 1)
        # 最近 4 轮 = 8 条，最早的问题1/回答1 被滑出窗口
        assert "问题1" not in out
        assert "回答1" not in out
        assert "问题5" in out

    @patch("routers.chat.get_redis")
    def test_history_empty_without_session(self, mock_redis):
        assert _load_recent_history("", 1) == ""
