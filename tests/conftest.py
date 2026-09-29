#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/conftest.py — pytest 全局 fixtures
"""
import os
import uuid
import pytest
from pathlib import Path
from config import reload_config


# ═══════════════════════════════════════════════════════════════
# 环境修复：覆盖 Windows 临时目录路径
# DSH 工具将 TMP/TEMP 锁定为独占目录，pytest 无法在其中创建子目录。
# 同时覆盖 PYTEST_DEBUG_TEMPROOT 以修复 pytest 内部缓存目录创建。
# ═══════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).parent.parent
_PYTEST_TMP_ROOT = str(_PROJECT_ROOT / ".temp_pip" / "pytest_tmp")
os.makedirs(_PYTEST_TMP_ROOT, exist_ok=True)
os.environ["TMP"]        = _PYTEST_TMP_ROOT
os.environ["TEMP"]       = _PYTEST_TMP_ROOT
os.environ["PYTEST_DEBUG_TEMPROOT"] = _PYTEST_TMP_ROOT
# 测试环境跳过配置启动校验（模型路径/API Key 等可能缺失）
os.environ["SKIP_CONFIG_VALIDATION"] = "1"


def _make_safe_tmpdir() -> Path:
    """
    创建一个安全可写的临时目录。
    使用 pathlib.Path.mkdir() 而非 tempfile.mkdtemp()，
    因为后者在 Windows 上以 mode=0o700 创建目录，会产生无法写入的 ACL。
    """
    base = Path(_PYTEST_TMP_ROOT)
    base.mkdir(parents=True, exist_ok=True)
    # 使用固定名称 tmp_path，与测试断言中期望的 source 值一致
    temp_dir = base / "tmp_path"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    return temp_dir


@pytest.fixture(autouse=True)
def _reload_config():
    """每个测试前重新加载配置，避免配置缓存影响测试结果。"""
    reload_config()


@pytest.fixture
def tmp_path():
    """
    覆盖 pytest 内置 tmp_path fixture。
    pytest 的 TempPathFactory 在 Windows 上会创建 mode=0o700 目录，
    导致写入文件时权限被拒（WinError 5）。
    此实现使用 pathlib.Path.mkdir() 创建默认 ACL 的临时目录，确保可写。
    """
    return _make_safe_tmpdir()


@pytest.fixture
def tmp_path_factory():
    """
    覆盖 pytest 内置 tmp_path_factory fixture。
    返回一个自定义 factory，使用 pathlib.Path.mkdir() 创建安全的临时目录。
    """
    class _SafeTmpPathFactory:
        def mktemp(self, *args, **kwargs):
            return _make_safe_tmpdir()

        def getbasetemp(self, *args, **kwargs):
            return Path(_PYTEST_TMP_ROOT)

    return _SafeTmpPathFactory()
