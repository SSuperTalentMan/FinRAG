#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
log_config.py — 日志配置
同时输出到控制台和文件，支持按大小轮转。

通过环境变量 LOG_FORMAT=json 切换为 JSON 结构化日志，便于 ELK / Loki 采集。
"""

import os
import sys
from pathlib import Path
from loguru import logger as _logger

from config import get_config


def setup_logging() -> None:
    """初始化日志：控制台 + 文件双输出。

    环境变量 LOG_FORMAT=json 时，文件日志切换为 loguru 原生 JSON 序列化格式，
    便于 ELK / Loki / Fluentd 采集与检索。
    """
    cfg = get_config()
    log_dir = Path(cfg.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger.remove()  # 移除默认 stdout handler

    use_json = os.environ.get("LOG_FORMAT", "").lower() == "json"

    # 控制台：INFO 及以上（始终用人类可读格式，方便开发调试）
    _logger.add(
        sys.stderr,
        level=cfg.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}",
        colorize=True,
    )

    # 文件：DEBUG，按 10MB 轮转，保留 7 天
    # LOG_FORMAT=json 时启用 loguru 原生 JSON 序列化（含异常栈、extra 字段）
    _logger.add(
        str(cfg.log_file),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        serialize=use_json,  # True 时输出 JSON 格式
        **({} if use_json else {"format": "{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function} | {message}"}),
    )


# 覆盖 stdlib logging 桥接
import logging as _stdlib_logging

class LoguruHandler(_stdlib_logging.Handler):
    """将标准 logging 桥接到 loguru。"""
    def emit(self, record):
        try:
            level = _logger.level(record.levelname).name if record.levelname else "INFO"
        except ValueError:
            level = record.levelname
        frame, depth = _stdlib_logging.currentframe(), 2
        while frame and frame.f_code.co_filename == __file__:
            frame, depth = frame.f_back, depth + 1
        _logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

_stdlib_logging.getLogger("uvicorn").addHandler(LoguruHandler())
_stdlib_logging.getLogger("fastapi").addHandler(LoguruHandler())
