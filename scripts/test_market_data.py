# -*- coding: utf-8 -*-
"""P5 实时行情注入复验：识别正负例 / 真实取数 / 缓存命中 / 断网静默降级。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import market_data as md  # noqa: E402

print("=" * 64)
print("A) 实时行情问句识别（正例应 True，负例应 False）")
print("=" * 64)
POS = [
    "现在的股市行情如何",
    "今天A股大盘表现怎么样",
    "上证指数现在多少点",
    "今天创业板指涨了多少",
    "最近股市走势如何",
    "沪深300收盘价是多少",
]
NEG = [
    "存款保险最高赔付多少",
    "什么是指数基金",
    "股票发行注册制改革的内容是什么",
    "创业板上市公司信息披露要求",
    "债券注册制改革全面落地了吗",
    "商业银行资本管理办法的核心要求",
    "毛利怎么算",  # 问数类
]
bad = 0
for q in POS:
    r = md.is_realtime_query(q)
    bad += (not r)
    print(f"  {'✓' if r else '✗ [应True]'} {q}")
for q in NEG:
    r = md.is_realtime_query(q)
    bad += r
    print(f"  {'✓' if not r else '✗ [应False]'} {q}")
print(f"识别准确: {len(POS) + len(NEG) - bad}/{len(POS) + len(NEG)}")

print()
print("=" * 64)
print("B) 真实取数（腾讯主源 / 东财备源）")
print("=" * 64)
t0 = time.time()
quotes = md.fetch_quotes(force=True)
t1 = time.time()
if quotes:
    for q in quotes:
        print(f"  {q['name']}({q['code']})  {q['price']}  涨跌 {q['change_amt']}  幅 {q['change_pct']}%  ts={q['quote_ts']}")
    print(f"  耗时 {t1 - t0:.2f}s")
else:
    print("  ✗ 取数为空（检查网络）")

print()
print("=" * 64)
print("C) 缓存命中（第二次调用应显著快于首次）")
print("=" * 64)
t2 = time.time()
md.fetch_quotes(force=False)
t3 = time.time()
print(f"  首次 {t1 - t0:.2f}s / 缓存 {t3 - t2:.4f}s -> {'✓ 命中缓存' if (t3 - t2) < (t1 - t0) else '? 未命中'}")

print()
print("=" * 64)
print("D) 上下文组装（LLM prompt 块）")
print("=" * 64)
ctx = md.build_market_context("现在的股市行情如何")
print(ctx if ctx else "  ✗ 空")

print()
print("=" * 64)
print("E) 断网静默降级（接口不可达时必须返回 None/空串，不抛异常）")
print("=" * 64)
from config import get_config  # noqa: E402


class _BadCfg:
    enabled = True
    provider = "tencent"            # 强制单源，避免备源把降级验证干扰掉
    tencent_url = "http://127.0.0.1:9/q="   # 保留端口，必然连接拒绝
    tencent_symbols = "sh000001"
    eastmoney_url = "http://127.0.0.1:9/nope"
    eastmoney_symbols = "1.000001"
    timeout_seconds = 2.0
    cache_ttl_seconds = 30


orig = md._cfg
md._cfg = lambda: _BadCfg()
md._cache.clear()
try:
    r = md.fetch_quotes(force=True)
    ctx2 = md.build_market_context("现在的股市行情如何")
    ok = (r is None) and (ctx2 == "")
    print(f"  fetch -> {r} | context -> {ctx2!r} | {'✓ 静默降级正常' if ok else '✗ 未按预期降级'}")
except Exception as e:  # noqa: BLE001
    print(f"  ✗ 抛异常（不可接受）: {type(e).__name__}: {e}")
finally:
    md._cfg = orig
    md._cache.clear()
