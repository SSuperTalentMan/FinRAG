#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/llm.py — LLM 服务（通义千问 / DeepSeek via DashScope）
提供流式和非流式两种调用模式。
"""

import json
import threading
from openai import OpenAI, AsyncOpenAI
from loguru import logger
from config import get_config
from services.metrics import record_llm
from rag_qa.core.prompts import RAGPrompts, sanitize_input

_client: OpenAI | None = None
_client_lock = threading.Lock()  # 双检锁：避免并发首请求重复创建客户端

_async_client: AsyncOpenAI | None = None
_async_client_lock = threading.Lock()  # 双检锁：异步客户端单例


def get_llm_client() -> OpenAI:
    """获取 DashScope 兼容的 OpenAI 客户端（单例，线程安全）。"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                cfg = get_config()
                _client = OpenAI(
                    api_key=cfg.api_key,
                    base_url=cfg.llm.base_url,
                    timeout=60,      # 单次请求超时（秒），避免 DashScope 挂起拖垮整个请求
                    max_retries=2,   # 瞬时失败自动重试
                )
                logger.info(f"LLM 客户端初始化: model={cfg.llm.model}")
    return _client


def get_async_llm_client() -> AsyncOpenAI:
    """获取异步 LLM 客户端（单例，线程安全）。

    用于 SSE 流式接口，避免 run_in_executor 逐 chunk 调度开销，
    实现真正的原生异步流式传输。
    """
    global _async_client
    if _async_client is None:
        with _async_client_lock:
            if _async_client is None:
                cfg = get_config()
                _async_client = AsyncOpenAI(
                    api_key=cfg.api_key,
                    base_url=cfg.llm.base_url,
                    timeout=60,
                    max_retries=2,
                )
                logger.info(f"异步 LLM 客户端初始化: model={cfg.llm.model}")
    return _async_client


def _create_with_fallback(messages: list[dict], stream: bool) -> tuple[object, str]:
    """调用主模型，失败且配置了备用模型时降级重试一次。

    返回 (response, 实际使用的模型名)。未配置备用模型或备用模型同样失败时抛出原异常。
    """
    cfg = get_config()
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=cfg.llm.model,
            messages=messages,
            stream=stream,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        return response, cfg.llm.model
    except Exception as primary_err:
        fallback = cfg.llm.fallback_model
        if not fallback or fallback == cfg.llm.model:
            raise
        logger.warning(f"主模型 {cfg.llm.model} 调用失败，降级备用模型 {fallback}: {primary_err}")
        response = client.chat.completions.create(
            model=fallback,
            messages=messages,
            stream=stream,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        return response, fallback


def chat_completion(messages: list[dict], stream: bool = False) -> str:
    """
    调用 LLM 生成回答。
    messages: [{"role": "user"/"assistant", "content": "..."}, ...]
    主模型失败时按配置自动降级备用模型（限流/模型下线等场景保证服务连续性）。
    """
    cfg = get_config()
    try:
        response, used_model = _create_with_fallback(messages, stream)
        if stream:
            # 流式：逐 token 拼接（流式响应无 usage 字段，仅记录调用次数）
            record_llm(used_model)
            full_text = ""
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
            return full_text
        else:
            # 非流式：响应携带 usage，记录 prompt/completion token 用量
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            record_llm(used_model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            if not response.choices:
                return ""
            return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise


def generate_answer(question: str, context: str, domain: str, history: str = "") -> str:
    """
    构造 Prompt 并调用 LLM 生成最终回答。
    Prompt 由 RAGPrompts.answer_messages 统一生成（system/user 分层，防注入），
    避免与流式接口重复定义。history 为对话历史文本，支持多轮上下文理解。
    """
    question = sanitize_input(question)
    messages = RAGPrompts.answer_messages(question, context, domain, history)
    return chat_completion(messages, stream=False)


def hyde_rewrite(question: str, domain: str) -> str:
    """
    HyDE（假设文档演化）策略：让 LLM 对问题进行假设性改写，生成更利于检索的查询。
    """
    prompt = f"""请将以下金融问题改写为一个假设性的专业陈述句，使其更易于在专业数据库中检索。保持金融专业术语。

原始问题：{question}
所属领域：{domain}

改写后的假设性陈述："""
    messages = [{"role": "user", "content": prompt}]
    return chat_completion(messages, stream=False)


def decompose_question(question: str, domain: str) -> list[str]:
    """
    子问题分解：将复杂问题拆分为多个独立子问题。
    """
    prompt = f"""请将以下{domain}领域的复杂金融问题分解为3-5个独立的子问题，每个子问题聚焦一个方面。
只输出子问题列表，每行一个，无需编号。

原始问题：{question}
所属领域：{domain}
"""
    messages = [{"role": "user", "content": prompt}]
    raw = chat_completion(messages, stream=False)
    # 解析多行输出
    sub_questions = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    return sub_questions[:5]  # 最多 5 个子问题
