FROM python:3.12-slim

WORKDIR /app

# 系统依赖（含 jieba / torch / rapidocr 等需要的编译与运行时库）
# default-mysql-client / redis-tools / curl：entrypoint.sh 与 compose healthcheck 依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev libgomp1 \
    libgl1 libglib2.0-0 \
    default-mysql-client redis-tools curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
# 注意：大型本地模型权重（bge-m3 / bge-reranker / bert）由 .dockerignore 排除，
# 运行时通过 docker-compose 的 volume 挂载进容器，避免镜像膨胀到数 GB。
COPY pyproject.toml uv.lock ./
# 先安装 uv（官方镜像不含），供 uv sync / uv run 使用
RUN pip install --no-cache-dir uv
RUN uv sync --no-dev

# 复制项目代码
COPY . .

# 创建日志目录并设置权限
RUN mkdir -p /app/logs && chown -R 1000:1000 /app

# 创建非 root 用户（UID/GID 1000），遵循最小权限原则
RUN groupadd -g 1000 finrag && useradd -m -u 1000 -g finrag finrag
USER finrag

EXPOSE 8000

ENTRYPOINT ["bash", "entrypoint.sh"]
