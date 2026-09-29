#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models.py — Pydantic 数据模型
定义 API 请求和响应的数据结构。
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─── 认证模型 ──────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)
    confirm_password: str = Field(..., min_length=6, max_length=128)
    captcha: str = Field(..., min_length=1, max_length=10, description="验证码")
    captcha_id: str = Field(..., min_length=1, max_length=128, description="验证码ID")
    email: str = Field("", max_length=100)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)
    captcha: str = Field(..., min_length=1, max_length=10)
    captcha_id: str = Field(..., min_length=1, max_length=128)


class CaptchaResponse(BaseModel):
    captcha_id: str
    captcha_image: str  # base64 编码的 PNG 图片


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str = "user"


class UserInfo(BaseModel):
    id: int
    username: str
    role: str
    email: str = ""
    avatar: str = ""
    status: str = "active"
    created_at: Optional[str] = None


class UserUpdateRequest(BaseModel):
    role: Optional[str] = Field(None, max_length=20)
    status: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=100)


# ─── 问答模型 ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """对话请求。"""
    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    domain: Optional[str] = Field(None, max_length=50, description="指定领域，None 则由意图识别自动判断")
    session_id: Optional[str] = Field(None, max_length=128, description="会话 ID，用于多轮对话")
    history: Optional[list[dict]] = Field(None, description="历史对话上下文")
    save_user: bool = Field(True, description="是否保存用户消息（重新生成时设为 false）")


class ChatResponse(BaseModel):
    """对话响应。"""
    answer: str
    domain: str
    confidence: float
    sources: list[dict] = Field(default_factory=list, description="来源文档列表")
    session_id: Optional[str] = None


class BM25Hit(BaseModel):
    question: str
    answer: str
    category: str
    score: float
    source: str = "mysql"


class MilvusHit(BaseModel):
    question: str
    answer: str
    category: str
    score: float
    source: str = "milvus"


class RerankHit(BaseModel):
    question: str
    answer: str
    category: str
    score: float
    source: str


# ─── 知识库模型 ────────────────────────────────────────────────────────────────
class KnowledgeBaseInfo(BaseModel):
    id: int
    name: str
    description: str
    domain: Optional[str]
    is_builtin: bool
    owner_id: Optional[int] = None  # 创建者用户 id；内置知识库为 None（系统属主）
    doc_count: int = 0
    created_at: Optional[str] = None


# ─── 通用响应 ──────────────────────────────────────────────────────────────────
class ApiResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: Optional[object] = None
