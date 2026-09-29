#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_llm.py — LLM 服务单元测试
"""
import pytest
from unittest.mock import patch, MagicMock
from services.llm import get_llm_client, chat_completion, generate_answer, hyde_rewrite, decompose_question


class TestLLMService:
    def test_get_llm_client_singleton(self):
        """get_llm_client 应返回单例。"""
        client1 = get_llm_client()
        client2 = get_llm_client()
        assert client1 is client2

    @patch('services.llm.get_llm_client')
    def test_chat_completion_structure(self, mock_client):
        """chat_completion 调用结构应正确。"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "测试回答"
        mock_response.choices[0].delta.content = None
        # 非流式响应携带 usage，供 token 用量指标记录
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = chat_completion([{"role": "user", "content": "你好"}])
        assert isinstance(result, str)
        assert result == "测试回答"
        mock_client.return_value.chat.completions.create.assert_called_once()

    @patch('services.llm.chat_completion')
    def test_hyde_rewrite_returns_string(self, mock_chat):
        """hyde_rewrite 应返回字符串。"""
        mock_chat.return_value = "假设性陈述：银行存款准备金是金融机构必须存放在中央银行的资金比例。"
        result = hyde_rewrite("什么是银行存款准备金", domain="banking")
        assert isinstance(result, str)
        mock_chat.assert_called_once()

    @patch('services.llm.chat_completion')
    def test_decompose_question_returns_string(self, mock_chat):
        """decompose_question 应返回列表。"""
        mock_chat.return_value = "子问题1\n子问题2\n子问题3"
        result = decompose_question("如何评估企业并购的财务影响", domain="corporate_finance")
        assert isinstance(result, list)
        assert len(result) == 3

    @patch('services.llm.chat_completion')
    def test_generate_answer_returns_string(self, mock_chat):
        """generate_answer 应返回字符串。"""
        mock_chat.return_value = "银行存款准备金是指..."
        result = generate_answer("测试问题", "测试上下文", "banking")
        assert isinstance(result, str)
