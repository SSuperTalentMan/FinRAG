#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_models.py — Pydantic 模型单元测试
"""
import pytest
from models import (
    RegisterRequest, LoginRequest, CaptchaResponse, TokenResponse,
    ChatRequest, ChatResponse, BM25Hit, MilvusHit, RerankHit,
    KnowledgeBaseInfo, ApiResponse,
)


class TestChatModels:
    def test_chat_request_valid(self):
        req = ChatRequest(message="你好")
        assert req.message == "你好"
        assert req.domain is None
        assert req.session_id is None

    def test_chat_request_with_all_fields(self):
        req = ChatRequest(
            message="测试",
            domain="banking",
            session_id="sess-001",
            history=[{"role": "user", "content": "hi"}],
        )
        assert req.domain == "banking"
        assert req.session_id == "sess-001"

    def test_chat_request_empty_message_rejected(self):
        with pytest.raises(Exception):
            ChatRequest(message="")

    def test_chat_response_default(self):
        resp = ChatResponse(answer="test", domain="general", confidence=0.0)
        assert resp.answer == "test"
        assert resp.sources == []
        assert resp.session_id is None

    def test_bm25_hit(self):
        hit = BM25Hit(question="q", answer="a", category="banking", score=0.85)
        assert hit.source == "mysql"

    def test_milvus_hit(self):
        hit = MilvusHit(question="q", answer="a", category="banking", score=0.9)
        assert hit.source == "milvus"

    def test_api_response_default(self):
        resp = ApiResponse()
        assert resp.success is True
        assert resp.message == "ok"
        assert resp.data is None

    def test_knowledge_base_info(self):
        kb = KnowledgeBaseInfo(id=1, name="测试库", description="desc", domain="banking", is_builtin=True)
        assert kb.doc_count == 0
        assert kb.created_at is None
