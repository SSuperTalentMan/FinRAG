#!/bin/bash
set -e

# 主机地址（默认连接宿主机已运行的 Redis / MySQL / Milvus）
MYSQL_HOST="${MYSQL_HOST:-host.docker.internal}"
MYSQL_PORT="${MYSQL_PORT:-3306}"
MYSQL_USER="${MYSQL_USER:-root}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-root}"
REDIS_HOST="${REDIS_HOST:-host.docker.internal}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
MILVUS_HOST="${MILVUS_HOST:-host.docker.internal}"
# Milvus 管理端口：HTTP 健康检查走 9091（19530 为 gRPC 服务端口，curl 无法探测）
MILVUS_HTTP_PORT="${MILVUS_HTTP_PORT:-9091}"
INIT_FLAG="/app/.initialized"

echo "=== FinRag 启动检查 ==="
echo "依赖服务: MySQL=${MYSQL_HOST}:${MYSQL_PORT} Redis=${REDIS_HOST}:${REDIS_PORT} Milvus=${MILVUS_HOST}:${MILVUS_HTTP_PORT}"

# 等待 MySQL
echo "等待 MySQL 就绪..."
until mysqladmin ping -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"${MYSQL_PASSWORD}" --silent 2>/dev/null; do
    echo "MySQL 未就绪，3s 后重试..."
    sleep 3
done
echo "MySQL 已就绪"

# 等待 Redis（支持密码）
REDIS_ARGS=()
if [ -n "$REDIS_PASSWORD" ]; then
    REDIS_ARGS+=("-a" "$REDIS_PASSWORD" "--no-auth-warning")
fi
echo "等待 Redis 就绪..."
until redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" "${REDIS_ARGS[@]}" ping >/dev/null 2>&1; do
    echo "Redis 未就绪，2s 后重试..."
    sleep 2
done
echo "Redis 已就绪"

# 等待 Milvus（HTTP 健康检查端口 9091 /healthz）
echo "等待 Milvus 就绪..."
for i in $(seq 1 30); do
    if curl -fsS "http://${MILVUS_HOST}:${MILVUS_HTTP_PORT}/healthz" >/dev/null 2>&1; then
        echo "Milvus 已就绪"
        break
    fi
    echo "Milvus 未就绪，${i}/30..."
    sleep 3
done

# 初始化数据（仅首次运行；数据目录为只读挂载，标记写到 /app 下）
if [ ! -f "$INIT_FLAG" ]; then
    echo "初始化数据库（首次运行）..."
    uv run python init_data.py
    touch "$INIT_FLAG"
fi

echo "=== 启动 FinRag 服务 ==="
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
