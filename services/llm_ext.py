#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/llm_ext.py — 多模态融合结构化/视觉 LLM 助手（移植自 DocAudit embeddings，统一收口 FinRag 配置）

提供 chat / chat_json / chat_json_validated / chat_vl 四类结构化调用：
- 强制 JSON 输出（response_format=json_object）+ 宽松 JSON 解析；
- 结构化输出加固：Pydantic Schema 校验失败回喂重试一次；
- 额度/模型不可用时按降级链就近切换并缓存；
- 思考模式显式关闭：抽取/比对/分级等确定性任务可把延迟降 10 倍以上，避免超时重试；
- trust_env=False：全链路防 HTTP_PROXY/HTTPS_PROXY 劫持本地请求。
Embedding 不再重复实现——复用 FinRag services/embedding（本地 BGE-M3）。
"""
from __future__ import annotations

import json
import logging

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from config import get_config
from services.metrics import record_llm

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# 对话模型降级链（按可用性就近切换）：sql_model 取配置或主 model
_CHAT_CANDIDATES = ["qwen3.6-plus", "qwen-plus-latest", "qwen-flash", "qwen-turbo", "qwen-max"]
# 视觉专用降级链：VL 严禁降级到纯文本模型（文本模型会剥掉图片只回敷衍文本）
_VL_CANDIDATES = ["qwen-vl-plus", "qwen2.5-vl-72b-instruct"]
_model_cache: dict[str, str] = {}


def _sql_model() -> str:
    cfg = get_config()
    return cfg.llm.sql_model or cfg.llm.model


def _fast_model() -> str:
    cfg = get_config()
    return cfg.llm.fast_model or cfg.llm.model


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        cfg = get_config()
        _client = AsyncOpenAI(
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            timeout=cfg.llm.timeout_seconds,
            max_retries=cfg.llm.max_retries,
            http_client=httpx.AsyncClient(trust_env=False, timeout=cfg.llm.timeout_seconds),
        )
    return _client


def _is_fallback_error(msg: str) -> bool:
    low = (msg or "").lower()
    return ("403" in low or "free quota" in low or "quota" in low
            or "model not found" in low or "invalidmodel" in low
            or "model does not exist" in low)


async def chat(
    messages: list[dict],
    model: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stage: str = "default",
    candidates: list[str] | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict]:
    """对话模型调用，额度/模型异常时就近降级并缓存；按 (model, stage) 计量 token。"""
    cfg = get_config()
    requested = model or _sql_model()
    chain = candidates if candidates is not None else _CHAT_CANDIDATES
    fallback = cfg.llm.fallback_model
    candidates_list = [requested] + [c for c in ([requested] + chain + ([fallback] if fallback else [])) if c != requested]
    last_err: Exception | None = None

    for m in candidates_list:
        kwargs: dict = {
            "model": m, "messages": messages,
            "temperature": cfg.llm.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # 思考模式显式关闭：qwen3/GLM 系列默认开启 thinking，确定性任务不需要推理链
        if m.startswith("qwen3"):
            kwargs["extra_body"] = {"enable_thinking": False}
        elif m.startswith("glm"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            resp = await get_client().chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_fallback_error(str(e)):
                logger.warning("对话模型 %s 不可用，尝试降级: %s", m, str(e)[:120])
                continue
            raise
        if m != requested:
            _model_cache[requested] = m
            logger.warning("对话模型降级: %s -> %s", requested, m)
        usage = {
            "tokens_in": getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0,
            "tokens_out": getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0,
        }
        record_llm(m, prompt_tokens=usage["tokens_in"], completion_tokens=usage["tokens_out"])
        content = resp.choices[0].message.content or ""
        return content, usage
    raise last_err or RuntimeError("无可用对话模型")


def _loads_lenient(content: str) -> dict:
    """宽松 JSON 解析：直接失败时先尝试截断修复，再截取首尾大括号（防模型输出包裹说明文字）。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # 截断修复：模型输出 JSON 未闭合（如「Unterminated string」）时，补齐括号/字符串后重试
    try:
        return json.loads(_repair_truncated_json(content))
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        return json.loads(content[start:end + 1])
    raise


def _repair_truncated_json(content: str) -> str:
    """尽力修复被截断的 JSON：闭合未完成的字符串，再按括号栈补齐 } / ]。

    仅在 json.loads 已失败后调用，不影响合法输出；用于根治问数生成「Unterminated string」类失败。
    """
    s = content.rstrip()
    if not s:
        return s
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    # 末尾若仍在中文字符串内（被截断），先闭合字符串
    if in_str:
        s += '"'
    # 去掉结尾可能残留的字段分隔符/冒号，避免闭合后仍是非法结构
    s = s.rstrip(" ,:")
    while stack:
        s += stack.pop()
    return s


async def chat_json(
    messages: list[dict], model: str | None = None,
    temperature: float | None = None, stage: str = "default",
) -> tuple[dict, dict]:
    content, usage = await chat(messages, model=model, json_mode=True, temperature=temperature, stage=stage)
    return _loads_lenient(content), usage


async def chat_json_validated(
    messages: list[dict], model: str | None, schema: type[BaseModel],
    temperature: float | None = None, stage: str = "default",
    max_tokens: int | None = None,
) -> tuple[BaseModel, dict]:
    """结构化输出加固：Pydantic 校验，失败把校验错误回喂模型重试一次。"""
    msgs = list(messages)
    last_err: Exception | None = None
    for attempt in range(2):
        content, usage = await chat(msgs, model=model, json_mode=True, temperature=temperature,
                                    stage=stage, max_tokens=max_tokens)
        try:
            return schema.model_validate(_loads_lenient(content)), usage
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            if attempt == 0:
                detail = getattr(e, "errors", lambda: str(e))()
                msgs = msgs + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                        f"上次输出未通过 JSON Schema 校验: {str(detail)[:400]}。"
                        "请修正后重新输出完整 JSON，不要添加任何解释文字。"},
                ]
    raise last_err or RuntimeError("structured output failed")


async def chat_vl(prompt: str, image_png: bytes, model: str | None = None,
                  stage: str = "vl") -> tuple[dict, dict]:
    """多模态调用（qwen-vl 系列）：图像 + 指令 → JSON。视觉模型间降级，耗尽抛错由调用方兜底。"""
    import base64

    cfg = get_config()
    data_url = "data:image/png;base64," + base64.b64encode(image_png).decode()
    content, usage = await chat(
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        model=model or cfg.llm.vl_model,
        json_mode=True,
        temperature=0.2,
        stage=stage,
        candidates=list(_VL_CANDIDATES),
    )
    return _loads_lenient(content), usage