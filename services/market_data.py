# -*- coding: utf-8 -*-
"""实时行情数据服务（P5 优化：行情类问句的实时数据注入）。

背景（真实观测到的问题）：
    知识库收录的是制度性/投教类静态资料，用户问「现在的股市行情如何」时，
    检索结果与问题语义相关度必然偏低（实测：重排后全部低于 0.7 阈值 →
    回退原始 Top-5），LLM 只能回答「背景资料未包含实时行情数据」。
    这不是检索 bug，而是「静态知识库被用来回答动态事实问题」的能力错配。

方案：
    在上下文构建阶段识别「实时行情类」问句，先取一份行情快照，作为独立
    上下文块（自带数据来源与数据时间说明）拼进 LLM prompt，并作为一条
    source 返回，让前端与用户能区分「实时数据」与「知识库制度性内容」。

设计约束（金融场景，安全优先）：
    1. 完全 fail-soft：任何网络/解析异常只记日志并返回空串，
       绝不抛异常、绝不影响主检索链路（行为等价于"未开启该功能"）；
    2. 短超时（默认 3s）+ 进程内 TTL 缓存（默认 30s）：
       休市时段、连续追问不会反复打接口；
    3. 代码不写死任何行情数值，只做「取数 → 格式化 → 标注时间与来源」；
    4. 数据源为公开行情接口，只读 GET，不涉及任何账户或交易能力。

日志约定：本模块用 loguru（与其他 services/* 一致），占位符一律 `{}`。
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from loguru import logger

from config import get_config

# 北京时间（行情接口返回的时间戳为 UTC+8 语义）
CST = timezone(timedelta(hours=8))

# ─── 行情类问句识别（规则，零开销）────────────────────────────────────────────
# 强信号：出现即判定为行情类问句
_STRONG_CUES = (
    "行情", "大盘", "点位", "涨跌幅", "涨停", "跌停", "实时报价", "最新报价",
    "收盘价", "开盘价", "股指", "走势", "涨了", "跌了", "涨多少", "跌多少",
)
# 主体词 + 时间/状态词 同时出现才判定（避免「什么是指数基金」这类知识型问句误触发）
_SUBJECT_CUES = (
    "股市", "a股", "沪指", "上证", "深证", "创业板", "沪深300", "科创50",
    "北证50", "科创板", "两市", "指数",
)
_TIME_STATE_CUES = (
    "现在", "今天", "今日", "目前", "最新", "实时", "当前", "近期", "最近",
    "多少点", "表现", "怎么样", "如何", "涨", "跌", "收盘", "开盘",
)


def is_realtime_query(question: str) -> bool:
    """判断问题是否为「实时行情类」（需要注入动态数据）。

    保守策略：宁可少注入，也不要在制度性问句里塞行情数据（污染上下文）。
    """
    if not question:
        return False
    q = question.lower()
    if any(c in q for c in _STRONG_CUES):
        return True
    has_subject = any(c in q for c in _SUBJECT_CUES)
    has_time_state = any(c in q for c in _TIME_STATE_CUES)
    return has_subject and has_time_state


# ─── 行情取数（多源 + TTL 缓存 + 失败短缓存）────────────────────────────────
# 主源用腾讯行情（urllib 实测可直连），备源用东方财富 push2。
# 注：东财 push2 对部分 urllib 客户端会直接断连（RemoteDisconnected），
# 因此"可切换 provider + auto 依次尝试"是必要设计，不能只挂一个源。
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, list[dict] | None]] = {}
# 失败结果缓存时长（秒）：避免离线时每次请求都白等一个超时
_FAIL_TTL = 10.0


def _cfg():
    return get_config().market_data


def _fetch_tencent(symbols: str, timeout: float) -> list[dict] | None:
    """腾讯行情：`v_sh000001="1~上证指数~000001~3888.11~昨收~今开~...~时间~涨跌额~涨跌幅~..."`"""
    url = f"{_cfg().tencent_url}{symbols}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("gbk", errors="replace")

    parsed: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or '="' not in line:
            continue
        body = line.split('="', 1)[1].rstrip('";').strip()
        f = body.split("~")
        if len(f) < 6:
            continue
        try:
            price = float(f[3])
        except (ValueError, IndexError):
            continue
        if price <= 0:          # 停牌/无行情，不展示占位符
            continue
        # 时间戳字段不固定位置（不同品种字段数不同）：按 14 位数字定位
        ts, amt, pct = 0, None, None
        for i, v in enumerate(f):
            if len(v) == 14 and v.isdigit():
                ts = int(datetime.strptime(v, "%Y%m%d%H%M%S").replace(tzinfo=CST).timestamp())
                for off, key in ((1, "amt"), (2, "pct")):
                    try:
                        val = float(f[i + off])
                    except (ValueError, IndexError):
                        continue
                    if key == "amt":
                        amt = val
                    else:
                        pct = val
                break
        parsed.append({
            "name": f[1].strip(), "code": f[2].strip(), "price": price,
            "change_amt": amt, "change_pct": pct, "quote_ts": ts,
        })
    return parsed or None


def _fetch_eastmoney(secids: str, timeout: float) -> list[dict] | None:
    """东方财富 push2：`fltt=2` 时 f2/f3/f4 为浮点直出，f124 为 UTC+8 秒级时间戳。"""
    cfg = _cfg()
    params = urllib.parse.urlencode({
        "fltt": 2,
        "secids": secids,
        "fields": "f1,f2,f3,f4,f12,f14,f124",
    })
    req = urllib.request.Request(f"{cfg.eastmoney_url}?{params}", headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8", errors="replace"))
    rows = ((raw or {}).get("data") or {}).get("diff") or []
    parsed: list[dict] = []
    for r in rows:
        price = r.get("f2")
        if not isinstance(price, (int, float)):   # 停牌时返回 "-"
            continue
        parsed.append({
            "name": str(r.get("f14") or "").strip(),
            "code": str(r.get("f12") or "").strip(),
            "price": float(price),
            "change_amt": r.get("f4") if isinstance(r.get("f4"), (int, float)) else None,
            "change_pct": r.get("f3") if isinstance(r.get("f3"), (int, float)) else None,
            "quote_ts": int(r.get("f124") or 0),
        })
    return parsed or None


_SOURCE_LABELS = {
    "tencent": "腾讯财经公开行情接口",
    "eastmoney": "东方财富公开行情接口",
}
_SOURCE_URLS = {
    "tencent": "https://gu.qq.com/",
    "eastmoney": "https://quote.eastmoney.com/",
}


def _fetch_any(*, force: bool = False) -> tuple[list[dict] | None, str]:
    """按 provider 取数，返回 (数据, 实际生效的数据源名称)。

    多源依次尝试是必要设计：东财 push2 会拒绝部分 urllib 客户端
    （RemoteDisconnected），单源挂掉不能让整个行情能力失效。
    """
    cfg = _cfg()
    providers = []
    for name in ([cfg.provider] if cfg.provider != "auto" else ["tencent", "eastmoney"]):
        if name == "tencent":
            providers.append(("tencent", cfg.tencent_symbols, _fetch_tencent))
        elif name == "eastmoney":
            providers.append(("eastmoney", cfg.eastmoney_symbols, _fetch_eastmoney))
    if not providers:
        logger.warning("行情 provider 配置无效: {}（可选 tencent/eastmoney/auto）", cfg.provider)
        return None, ""

    cache_key = "|".join(f"{n}:{s}" for n, s, _ in providers)
    now = time.time()
    if not force:
        with _cache_lock:
            hit = _cache.get(cache_key)
        if hit:
            ts, data, src = hit
            ttl = cfg.cache_ttl_seconds if data else _FAIL_TTL
            if now - ts < ttl:
                return data, src

    data: list[dict] | None = None
    source = ""
    for name, symbols, fn in providers:
        try:
            data = fn(symbols, cfg.timeout_seconds)
        except Exception as e:  # noqa: BLE001 — 行情是增强项，任何失败都必须静默
            logger.warning("行情源 {} 调用失败（尝试下一源）: {}: {}",
                           name, type(e).__name__, str(e)[:110])
            continue
        if data:
            source = _SOURCE_LABELS.get(name, name)
            logger.info("行情源 {} 取数成功，{} 条", name, len(data))
            break

    with _cache_lock:
        _cache[cache_key] = (now, data, source)
    return data, source


def fetch_quotes(*, force: bool = False) -> list[dict] | None:
    """取主要指数快照（失败返回 None，调用方需容忍）。"""
    return _fetch_any(force=force)[0]



# ─── 格式化 ──────────────────────────────────────────────────────────────────
def _session_note(now: datetime) -> str:
    """区分交易时段/非交易时段——休市时明确告知"这是最近交易日收盘值"。"""
    if now.weekday() >= 5:
        return "当前为休市日（周末），以上为最近一个交易日的收盘数据"
    t = now.hour * 60 + now.minute
    if 9 * 60 + 15 <= t <= 15 * 60 + 5:
        return "当前为交易时段，以上为盘中快照数据（公开接口可能有秒级延迟）"
    if t < 9 * 60 + 15:
        return "当前为盘前时段，以上为最近一个交易日的收盘数据"
    return "当前为盘后时段，以上为最近一个交易日的收盘数据"


def _fmt_row(item: dict) -> str:
    name, code, price = item["name"], item["code"], item["price"]
    amt, pct = item.get("change_amt"), item.get("change_pct")
    if pct is None:
        return f"- {name}（{code}）：{price}"
    direction = "涨" if pct > 0 else ("跌" if pct < 0 else "平")
    if pct == 0:
        return f"- {name}（{code}）：{price}，持平"
    amt_txt = f"{abs(amt):.2f}" if isinstance(amt, (int, float)) else "—"
    return f"- {name}（{code}）：{price}，{direction} {amt_txt}，{direction}幅 {abs(pct):.2f}%"


def build_market_context(question: str = "") -> str:
    """组装行情上下文块；失败返回空串（主链路照旧走知识库检索）。"""
    quotes, source = _fetch_any()
    if not quotes:
        return ""

    quote_dt = None
    for q in quotes:
        if q.get("quote_ts"):
            quote_dt = datetime.fromtimestamp(q["quote_ts"], CST)
            break
    quote_time = quote_dt.strftime("%Y-%m-%d %H:%M:%S") if quote_dt else "时间未知"

    lines = [
        f"【实时行情数据】（数据时间：{quote_time}；来源：{source or '公开行情接口'}）",
        *[_fmt_row(q) for q in quotes],
        f"说明：{_session_note(datetime.now(CST))}。",
        "回答要求：行情/涨跌/点位类问题请优先依据以上实时数据作答，并注明数据时间；",
        "知识库检索到的制度性、投教类内容属于静态资料，请与实时行情区分表述，不得相互混淆。",
    ]
    return "\n".join(lines)


def last_snapshot_meta() -> dict:
    """最近一次成功取数的溯源信息（供 sources 标注，避免写死数据源）。"""
    with _cache_lock:
        for key, (_, data, src) in _cache.items():
            if not data:
                continue
            ts = next((q["quote_ts"] for q in data if q.get("quote_ts")), 0)
            quote_time = datetime.fromtimestamp(ts, CST).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
            provider = key.split(":", 1)[0]
            return {
                "source": src or "公开行情接口",
                "source_url": _SOURCE_URLS.get(provider, ""),
                "quote_time": quote_time,
            }
    return {"source": "公开行情接口", "source_url": "", "quote_time": ""}


def snapshot_stats() -> dict:
    """可观测：缓存条目数与最近一次取数结果（供 /health/ready 或诊断脚本使用）。"""
    with _cache_lock:
        items = list(_cache.items())
    return {
        "cache_entries": len(items),
        "cache_keys": [k for k, _, _ in items],
        "last_ok": any(v for _, v, _ in items),
    }
