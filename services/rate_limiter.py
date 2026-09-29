#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services/rate_limiter.py — Redis 分布式限流工具

固定窗口计数（INCR + EXPIRE，原子操作），天然支持多实例部署：
相比进程内内存限流，限流状态可跨 worker / 多副本共享，杜绝单进程窗口被绕过。

Redis 不可用时 fail-open（放行 + 告警），避免限流组件故障拖垮业务；
认证接口的防爆破仍有图形验证码作为主防线。
"""

import time
from loguru import logger
from db.redis import get_redis

RL_PREFIX = "finrag:rl:"


def check_rate_limit(key: str, max_count: int, window_seconds: int) -> tuple[bool, int]:
    """
    固定窗口限流。返回 (allowed, current_count)。

    key 为限流维度（如 "ip:1.2.3.4" / "user:12"）；
    窗口按 window_seconds 对齐分片，同一分片内的请求共享计数。
    """
    try:
        r = get_redis()
        slot = int(time.time()) // window_seconds
        rk = f"{RL_PREFIX}{key}:{slot}"
        count = r.incr(rk)
        if count == 1:
            # 多给 1s 余量，避免分片边界处 key 被提前清理导致计数归零
            r.expire(rk, window_seconds + 1)
        return count <= max_count, int(count)
    except Exception as e:
        logger.warning(f"限流 Redis 异常，放行（fail-open）: {e}")
        return True, 0
