#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rag_qa/mcp_layer/registry.py — MCP 工具管控层（纯标准库，零新增依赖）。

四件事缺一不可：未知工具拒绝 → 身份越权拒绝 → 参数校验 → 执行（线程池 + 超时）→ 审计。

设计要点（均为实战踩坑结论）：
1. 参数校验只做 必填/类型/非空/enum 四类；bool 是 int 子类，integer/number 校验要先排除。
2. 超时保护用 ThreadPoolExecutor(max_workers=1)，超时后 shutdown(wait=False) 不等卡住的线程，
   否则超时保护形同虚设。
3. 审计脱敏必须「字段名 + 值模式」双重：参数名常是 question/code 这类泛化命名，
   只按字段名匹配会漏；值模式正则兜住身份证/手机号/长密钥。
4. 审计写 JSONL，best-effort —— 落盘失败绝不影响主流程。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

# ── 数据结构 ───────────────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    """工具统一返回。ok=False 时 error 必须给人能看懂的原因（不 mock、不静默）。"""
    ok: bool
    data: Any = None
    error: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"ok": self.ok, "meta": self.meta}
        if self.ok:
            d["data"] = self.data
        else:
            d["error"] = self.error
        return d


@dataclass
class Tool:
    name: str                  # 点号分层，如 finrag.kb_search
    description: str
    input_schema: dict         # JSON Schema 子集：type/properties/required
    handler: Callable[[dict], ToolResult]
    scopes: tuple = ("agent",)  # 允许调用的 caller 身份白名单
    timeout: float = 60.0

    def to_mcp_spec(self) -> dict:
        """转 MCP tools/list 的工具描述格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


# ── 审计脱敏：字段名 + 值模式双重 ─────────────────────────────────────────────
_SENSITIVE_KEYS = ("password", "token", "secret", "api_key", "apikey", "authorization")
_VALUE_PATTERNS = (
    re.compile(r"^\d{17}[\dXx]$"),            # 身份证
    re.compile(r"^1\d{10}$"),                 # 手机号
    re.compile(r"^(sk|fk)-[A-Za-z0-9]{16,}$"),  # 常见 API Key 形态
    re.compile(r"^[A-Za-z0-9_\-.]{40,}$"),    # 超长疑似密钥串（JWT 等）
)
_MASK = "***"


def mask_args(args: dict) -> dict:
    """审计前脱敏：命中敏感字段名或值模式的一律打码。"""
    out: dict = {}
    for k, v in (args or {}).items():
        if any(s in k.lower() for s in _SENSITIVE_KEYS):
            out[k] = _MASK
            continue
        if isinstance(v, str) and any(p.match(v) for p in _VALUE_PATTERNS):
            out[k] = _MASK
            continue
        if isinstance(v, str) and len(v) > 300:
            v = v[:300] + "…"  # 长文本截断，审计只留摘要
        out[k] = v
    return out


# ── 参数校验（必填 / 类型 / 非空 / enum）───────────────────────────────────────
def _validate(args: dict, schema: dict) -> str | None:
    """返回 None 表示通过，否则返回可直接回喂 LLM 的错误说明。"""
    props = schema.get("properties", {})
    for req in schema.get("required", []) or []:
        v = args.get(req)
        if v is None or (isinstance(v, str) and not v.strip()):
            return f"缺少必填参数: {req}"
    for k, v in args.items():
        spec = props.get(k)
        if spec is None:
            continue  # 未声明的参数不校验（宽松透传，由 handler 自行把关）
        want = spec.get("type")
        if want in ("integer", "number"):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return f"参数 {k} 应为 {want}"
        elif want == "string":
            if not isinstance(v, str):
                return f"参数 {k} 应为 string"
            if not v.strip():
                return f"参数 {k} 不能为空"
        elif want == "boolean":
            if not isinstance(v, bool):
                return f"参数 {k} 应为 boolean"
        if "enum" in spec and v not in spec["enum"]:
            return f"参数 {k} 必须是 {'/'.join(map(str, spec['enum']))}"
    return None


# ── 注册表 ────────────────────────────────────────────────────────────────────
class ToolRegistry:
    def __init__(self, audit_path: str = ""):
        self._tools: dict[str, Tool] = {}
        self._audit_path = audit_path or os.environ.get(
            "FINRAG_MCP_AUDIT",
            os.path.join("logs", "mcp_audit.jsonl"),
        )
        self._audit_lock = threading.Lock()

    # -- 注册与查询 --
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [t.to_mcp_spec() for t in self._tools.values()]

    # -- 调用（管控主入口）--
    def call(self, name: str, args: dict | None, caller: str = "agent") -> ToolResult:
        args = args or {}
        tool = self._tools.get(name)
        if tool is None:
            return self._finish(name, caller, args, ToolResult(False, error=f"未知工具: {name}"), 0.0)

        if caller not in tool.scopes:
            return self._finish(name, caller, args, ToolResult(False, error=f"身份 {caller} 无权调用 {name}"), 0.0)

        err = _validate(args, tool.input_schema)
        if err:
            return self._finish(name, caller, args, ToolResult(False, error=f"参数校验失败: {err}"), 0.0)

        # 线程池超时隔离：超时后不等卡住的线程（否则超时保护形同虚设）
        started = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(tool.handler, args)
            try:
                result = future.result(timeout=tool.timeout)
            except TimeoutError:
                result = ToolResult(
                    False, error=f"工具 {name} 执行超时（>{tool.timeout}s），请缩小问题范围后重试",
                    meta={"timeout": tool.timeout},
                )
            except Exception as e:  # noqa: BLE001 handler 内部异常统一收口
                result = ToolResult(False, error=f"工具 {name} 执行异常: {type(e).__name__}: {str(e)[:200]}")
        finally:
            pool.shutdown(wait=False)
        elapsed = time.monotonic() - started
        return self._finish(name, caller, args, result, elapsed)

    # -- 审计（best-effort）--
    def _finish(self, name: str, caller: str, args: dict, result: ToolResult, elapsed: float) -> ToolResult:
        result.meta.setdefault("tool", name)
        result.meta.setdefault("caller", caller)
        result.meta.setdefault("elapsed_ms", int(elapsed * 1000))
        self._audit(name, caller, args, result, elapsed)
        return result

    def _audit(self, name: str, caller: str, args: dict, result: ToolResult, elapsed: float) -> None:
        try:
            os.makedirs(os.path.dirname(self._audit_path) or ".", exist_ok=True)
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tool": name,
                "caller": caller,
                "args": mask_args(args),
                "ok": result.ok,
                "error": result.error[:200],
                "elapsed_ms": int(elapsed * 1000),
            }
            with self._audit_lock:
                with open(self._audit_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 审计失败绝不影响主流程
            pass
