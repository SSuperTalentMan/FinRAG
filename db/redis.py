#!/usr/bin/env python
"""
db/redis.py — Redis 连接与缓存工具
"""

import hashlib
import json
import time

import redis
from loguru import logger

from config import get_config
from services.metrics import record_qa_cache

# ─── 缓存 key 前缀 ─────────────────────────────────────────────────────────────
CACHE_PREFIX = "finrag:qa:"

# ─── 会话存储（单一事实来源：chat / conversation / admin 路由统一从这里导入）──
SESSION_KEY_PREFIX = "finrag:session:"    # finrag:session:{session_id}
# value: JSON string -> {"user_id": int, "created_at": int,
#                        "messages": [{"role", "content", "domain", "ts"}, ...]}
SESSION_TTL = 86400 * 7                   # 7天过期
SESSION_MAX_MESSAGES = 50                 # 每会话保留的最大消息数

# 会话索引：finrag:user_sessions:{uid} -> ZSET(member=session_id, score=最近活跃 ts)
# 用于会话列表/O(N)扫描优化；TTL 略长于会话本体，随会话自然过期
SESSION_INDEX_PREFIX = "finrag:user_sessions:"
SESSION_INDEX_TTL = SESSION_TTL + 86400

# 进程级单例客户端（redis-py 客户端线程安全且内置连接池，避免每次调用新建连接池导致连接泄漏）
_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """获取 Redis 连接实例（Singleton，内部自带连接池）。"""
    global _redis_client
    if _redis_client is None:
        cfg = get_config()
        _redis_client = redis.Redis(
            host=cfg.redis.host,
            port=cfg.redis.port,
            password=cfg.redis.password or None,
            db=cfg.redis.db,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30,
        )
        logger.debug("Redis 单例客户端已初始化")
    return _redis_client


def close_redis() -> None:
    """应用关闭时释放 Redis 连接（优雅停机）。"""
    global _redis_client
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception as e:
            logger.warning(f"关闭 Redis 连接失败: {e}")
        _redis_client = None


def make_cache_key(text: str) -> str:
    """根据文本生成缓存 key（md5 前缀）。"""
    raw = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{CACHE_PREFIX}{raw}"


def cache_get(text: str):
    """查询缓存，命中返回解析后的 dict，否则返回 None。"""
    try:
        r = get_redis()
        key = make_cache_key(text)
        val = r.get(key)
        if val:
            data = json.loads(val)
            logger.debug(f"Redis 缓存命中: {text[:30]}...")
            record_qa_cache(hit=True)
            return data
    except Exception as e:
        logger.warning(f"Redis GET 失败: {e}")
    record_qa_cache(hit=False)
    return None


def cache_set(text: str, data: dict, ttl: int = 86400) -> None:
    """写入缓存。"""
    try:
        r = get_redis()
        key = make_cache_key(text)
        r.setex(key, ttl, json.dumps(data, ensure_ascii=False))
        logger.debug(f"Redis 缓存写入: {text[:30]}... TTL={ttl}s")
    except Exception as e:
        logger.warning(f"Redis SET 失败: {e}")


def cache_delete(text: str) -> None:
    """删除缓存。"""
    try:
        r = get_redis()
        r.delete(make_cache_key(text))
    except Exception as e:
        logger.warning(f"Redis DELETE 失败: {e}")


# ─── 会话消息原子追加（Lua）───────────────────────────────────────────────────
# 读取-追加-裁剪-写回在 Redis 服务端一次完成，杜绝并发读改写互相覆盖
# （如"重新生成"与新的流式回答同时写同一会话时丢失消息）。
# cjson 按字节处理字符串，UTF-8 中文原样保留，与 json.loads 的读取方兼容。
_APPEND_MSG_LUA = """
local raw = redis.call('GET', KEYS[1])
local data = nil
if raw then
    local ok, decoded = pcall(cjson.decode, raw)
    if ok then data = decoded end
end
if not data or type(data.messages) ~= 'table' then
    data = { messages = {} }
    if ARGV[4] ~= '' then data['user_id'] = tonumber(ARGV[4]) end
    data['created_at'] = tonumber(ARGV[5]) or 0
end
table.insert(data.messages, cjson.decode(ARGV[2]))
local maxn = tonumber(ARGV[3])
while #data.messages > maxn do
    table.remove(data.messages, 1)
end
redis.call('SETEX', KEYS[1], tonumber(ARGV[1]), cjson.encode(data))
return #data.messages
"""

# (client, script) 缓存：客户端单例重建后需重新 register_script
_append_script_cache: tuple | None = None


def append_session_message(
    session_id: str,
    message: dict,
    user_id: int | None = None,
    max_messages: int = SESSION_MAX_MESSAGES,
    ttl: int = SESSION_TTL,
) -> int:
    """原子追加一条会话消息，返回追加后的消息总数。

    Lua 脚本执行失败时降级为读改写（非原子，仅作兜底）；
    Redis 整体不可用时异常向上抛出，由调用方决定 fail-open 或报错。
    """
    global _append_script_cache
    r = get_redis()
    key = f"{SESSION_KEY_PREFIX}{session_id}"
    try:
        if _append_script_cache is None or _append_script_cache[0] is not r:
            _append_script_cache = (r, r.register_script(_APPEND_MSG_LUA))
        count = int(_append_script_cache[1](
            keys=[key],
            args=[
                ttl,
                json.dumps(message, ensure_ascii=False),
                max_messages,
                user_id if user_id is not None else "",
                int(time.time()),
            ],
        ))
        if user_id is not None:
            index_session(user_id, session_id, int(message.get("ts") or time.time()))
        return count
    except Exception as lua_err:
        logger.warning(f"Lua 原子追加失败，降级读改写: {lua_err}")

    # ── 兜底：读改写（与 Lua 相同语义，但非原子）──
    raw = r.get(key)
    if raw:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
    else:
        data = None
    if not data or not isinstance(data.get("messages"), list):
        data = {"messages": [], "created_at": int(time.time())}
        if user_id is not None:
            data["user_id"] = user_id
    data["messages"].append(message)
    data["messages"] = data["messages"][-max_messages:]
    r.setex(key, ttl, json.dumps(data, ensure_ascii=False))
    if user_id is not None:
        index_session(user_id, session_id, int(message.get("ts") or time.time()))
    return len(data["messages"])


# ─── 会话索引（按用户）────────────────────────────────────────────────────────
def index_session(user_id: int, session_id: str, ts: int) -> None:
    """登记/刷新会话索引（ZSET），fail-open：索引缺失只影响列表性能，不影响正确性。"""
    try:
        r = get_redis()
        idx_key = f"{SESSION_INDEX_PREFIX}{user_id}"
        r.zadd(idx_key, {session_id: int(ts or time.time())})
        r.expire(idx_key, SESSION_INDEX_TTL)
    except Exception as e:
        logger.warning(f"会话索引更新失败: {e}")


def unindex_session(user_id: int, session_id: str) -> None:
    """从会话索引移除（清空会话/删除用户时调用）。"""
    try:
        get_redis().zrem(f"{SESSION_INDEX_PREFIX}{user_id}", session_id)
    except Exception as e:
        logger.warning(f"会话索引移除失败: {e}")


def _scan_user_sessions(r: redis.Redis, uid: int, rebuild_index: bool = True) -> list[tuple[str, dict]]:
    """回退路径：scan 全量会话 key 并过滤属主；顺带重建该用户的 ZSET 索引。"""
    keys = list(r.scan_iter(f"{SESSION_KEY_PREFIX}*"))
    if not keys:
        return []
    pipe = r.pipeline()
    for key in keys:
        pipe.get(key)
    values = pipe.execute()

    result: list[tuple[str, dict]] = []
    idx_key = f"{SESSION_INDEX_PREFIX}{uid}"
    indexed = False
    for key, val in zip(keys, values):
        if not val:
            continue
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        try:
            data = json.loads(val)
        except json.JSONDecodeError:
            continue
        if data.get("user_id") != uid:
            continue
        sid = key.replace(SESSION_KEY_PREFIX, "", 1) if isinstance(key, str) \
            else key.decode("utf-8").replace(SESSION_KEY_PREFIX, "", 1)
        result.append((sid, data))
        if rebuild_index:
            msgs = data.get("messages", [])
            last_ts = max((m.get("ts", 0) for m in msgs), default=0) if msgs else data.get("created_at", 0)
            try:
                r.zadd(idx_key, {sid: int(last_ts or data.get("created_at", 0))})
                indexed = True
            except Exception:
                pass
    if indexed:
        try:
            r.expire(idx_key, SESSION_INDEX_TTL)
        except Exception:
            pass
    return result


def get_user_sessions(uid: int) -> list[tuple[str, dict]]:
    """获取某用户的全部活跃会话，按最近活跃降序返回 [(session_id, data)]。

    优先走 ZSET 索引（pipeline 批量取值，1 次往返）；索引为空（存量数据/索引丢失）
    时回退 scan 全量并顺带重建索引。已过期/损坏的会话条目会从索引中惰性剔除。
    """
    r = get_redis()
    idx_key = f"{SESSION_INDEX_PREFIX}{uid}"
    try:
        ids = r.zrevrange(idx_key, 0, -1)
    except Exception as e:
        logger.warning(f"会话索引读取失败（回退 scan）: {e}")
        ids = []

    if not ids:
        return _scan_user_sessions(r, uid)

    pipe = r.pipeline()
    for sid in ids:
        pipe.get(f"{SESSION_KEY_PREFIX}{sid}")
    values = pipe.execute()

    result: list[tuple[str, dict]] = []
    missing: list[str] = []
    for sid, val in zip(ids, values):
        if not val:
            missing.append(sid)  # 会话已过期：惰性清理索引
            continue
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        try:
            data = json.loads(val)
        except json.JSONDecodeError:
            missing.append(sid)
            continue
        result.append((sid, data))
    if missing:
        try:
            r.zrem(idx_key, *missing)
        except Exception:
            pass
    return result


def clear_qa_cache() -> int:
    """清空所有 QA 回答缓存（文档/知识库变更后调用，避免旧答案残留最多 24h）。

    缓存 key 前缀为 CACHE_PREFIX（finrag:qa:）。返回清除的 key 数量；失败返回 0。
    """
    try:
        r = get_redis()
        keys = list(r.scan_iter(f"{CACHE_PREFIX}*"))
        if keys:
            deleted = r.delete(*keys)
            logger.info(f"已清空 QA 缓存: {deleted} 条")
            return int(deleted)
    except Exception as e:
        logger.warning(f"清空 QA 缓存失败: {e}")
    return 0
