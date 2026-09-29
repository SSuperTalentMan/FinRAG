#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
routers/auth.py — 用户认证路由
提供注册、登录、验证码生成接口，使用 JWT 签发 Token。
密码使用 bcrypt 哈希存储，验证码使用 Pillow 生成随机图形验证码。
"""

import io
import base64
import secrets
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from models import (
    RegisterRequest, LoginRequest, CaptchaResponse,
    TokenResponse, ApiResponse, UserInfo,
)
from db.redis import get_redis
from db.mysql import (
    create_user, verify_password, get_user_by_username, get_user_by_id,
)
from services.audit import record_audit

router = APIRouter(prefix="/auth", tags=["认证"])

security = HTTPBearer(auto_error=False)

# ─── JWT 配置 ──────────────────────────────────────────────────────────────────
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24
_JWT_SECRET_FILE = Path(__file__).resolve().parent.parent / ".jwt_secret"


def _load_or_create_secret() -> str:
    """加载或创建持久化的 JWT Secret，确保服务重启后旧 Token 仍然有效。"""
    if _JWT_SECRET_FILE.exists():
        return _JWT_SECRET_FILE.read_text().strip()
    key = secrets.token_hex(32)
    _JWT_SECRET_FILE.write_text(key)
    return key


JWT_SECRET_KEY = _load_or_create_secret()


def _create_token(user_id: int, username: str, role: str) -> str:
    """签发 JWT Token。"""
    import jose.jwt as jwt
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "uid": user_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _verify_token(token: str) -> dict:
    """验证 JWT Token，返回 {username, user_id, role}。"""
    import jose.jwt as jwt
    from jose import JWTError
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {
            "username": payload["sub"],
            "user_id": payload.get("uid"),
            "role": payload.get("role", "user"),
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """从请求头中提取并验证用户，返回用户信息字典。

    JWT 签名有效 ≠ 账号可用：用户可能已被删除 / 禁用 / 降级，而 Token 最长 24h 有效。
    这里回查 DB 实时校验状态与角色，保证管理操作（禁用、降级、删除）即时生效，
    避免"被禁用用户仍可问答、被降级管理员仍可调管理接口"的越权窗口。
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="未提供认证信息")
    claims = _verify_token(credentials.credentials)

    user = get_user_by_id(claims["user_id"]) if claims.get("user_id") is not None else None
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在或已被删除")
    if user.get("status") == "disabled":
        raise HTTPException(status_code=403, detail="账号已被禁用")

    # 角色以 DB 为准（Token 中的 role 可能已过期，如管理员被降级）
    return {
        "username": user["username"],
        "user_id": user["id"],
        "role": user.get("role", "user"),
    }


def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """验证当前用户是否为管理员。"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user


# ─── 图形验证码生成 ─────────────────────────────────────────────────────────────
CAPTCHA_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CAPTCHA_LENGTH = 4
CAPTCHA_WIDTH = 120
CAPTCHA_HEIGHT = 48


def generate_captcha_image(text: str) -> str:
    """生成图形验证码，返回 base64 PNG。

    清晰度优先：加粗大号字体、深色高对比字符，
    干扰元素数量少且颜色极浅，保证肉眼易读。
    """
    img = Image.new("RGB", (CAPTCHA_WIDTH, CAPTCHA_HEIGHT), color="#ffffff")
    draw = ImageDraw.Draw(img)

    # 优先使用粗体字体，提升可读性
    font = None
    for fname in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            font = ImageFont.truetype(fname, 30)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    for i, ch in enumerate(text):
        offset = random.randint(-2, 2)
        x = 15 + i * 26 + offset
        y = random.randint(8, 12) + offset
        color = (random.randint(0, 70), random.randint(0, 70), random.randint(0, 70))
        draw.text((x, y), ch, fill=color, font=font)

    # 干扰线：仅 2 条且颜色极浅，不遮挡字符
    for _ in range(2):
        draw.line(
            [
                (random.randint(0, CAPTCHA_WIDTH), random.randint(0, CAPTCHA_HEIGHT)),
                (random.randint(0, CAPTCHA_WIDTH), random.randint(0, CAPTCHA_HEIGHT)),
            ],
            fill=(random.randint(230, 245), random.randint(230, 245), random.randint(230, 245)),
            width=1,
        )

    # 干扰点：少量、极浅
    for _ in range(8):
        draw.point(
            (random.randint(0, CAPTCHA_WIDTH), random.randint(0, CAPTCHA_HEIGHT)),
            fill=(random.randint(200, 235), random.randint(200, 235), random.randint(200, 235)),
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── 路由接口 ──────────────────────────────────────────────────────────────────

@router.get("/captcha", response_model=CaptchaResponse, summary="获取验证码")
def get_captcha():
    """生成随机验证码，存入 Redis（5分钟有效期），返回 base64 图片。"""
    text = "".join(random.choices(CAPTCHA_CHARS, k=CAPTCHA_LENGTH))
    captcha_id = secrets.token_hex(8)

    try:
        r = get_redis()
        r.setex(f"finrag:captcha:{captcha_id}", 300, text.lower())
    except Exception as e:
        logger.error(f"验证码写入 Redis 失败: {e}")
        raise HTTPException(status_code=503, detail="验证码服务暂不可用，请稍后重试")

    img_b64 = generate_captcha_image(text)
    logger.debug(f"验证码生成: id={captcha_id}, code={text}")
    return CaptchaResponse(captcha_id=captcha_id, captcha_image=f"data:image/png;base64,{img_b64}")


def _verify_captcha(captcha_id: str, captcha_input: str) -> bool:
    """验证验证码是否正确，验证通过后删除。Redis 异常时失败关闭（拒绝放行）。"""
    try:
        r = get_redis()
        key = f"finrag:captcha:{captcha_id}"
        stored = r.get(key)
        if not stored:
            return False
        stored_code = stored.decode("utf-8").lower() if isinstance(stored, bytes) else stored.lower()
        if stored_code != captcha_input.lower():
            return False
        r.delete(key)
        return True
    except Exception as e:
        logger.warning(f"验证码校验异常（按失败处理）: {e}")
        return False


@router.post("/register", response_model=ApiResponse, summary="用户注册")
def register(req: RegisterRequest, request: Request):
    """注册新用户。"""
    # 验证密码
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="两次密码不一致")

    # 验证验证码
    if not _verify_captcha(req.captcha_id, req.captcha):
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    # 检查用户名是否已存在
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="用户名已存在")

    # 创建用户（并发重名时靠 username 唯一索引兜底，避免检查-插入竞态导致 500）
    try:
        user_id = create_user(req.username, req.password, req.email, role="user")
    except Exception as e:
        if "Duplicate entry" in str(e) or (hasattr(e, "args") and e.args and e.args[0] == 1062):
            raise HTTPException(status_code=400, detail="用户名已存在")
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(status_code=500, detail="注册失败，请稍后重试")
    record_audit(user_id, req.username, "auth.register", detail="新用户注册", request=request)
    logger.info(f"用户注册成功: id={user_id}, username={req.username}")
    return ApiResponse(success=True, message="注册成功，请登录")


@router.post("/login", response_model=ApiResponse, summary="用户登录")
def login(req: LoginRequest, request: Request):
    """用户登录，验证用户名/密码后签发 JWT。"""
    logger.info(f"用户登录请求: username={req.username}")

    # 验证验证码
    if not _verify_captcha(req.captcha_id, req.captcha):
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    # 验证密码
    user = verify_password(req.username, req.password)
    if not user:
        record_audit(None, req.username, "auth.login_failed", detail="用户名或密码错误",
                     request=request, success=False)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if user.get("status") == "disabled":
        record_audit(user["id"], user["username"], "auth.login_failed", detail="账号已被禁用",
                     request=request, success=False)
        raise HTTPException(status_code=403, detail="账号已被禁用")

    token = _create_token(user["id"], user["username"], user.get("role", "user"))
    record_audit(user["id"], user["username"], "auth.login", request=request)
    logger.info(f"登录成功: username={req.username}, role={user.get('role')}")
    return ApiResponse(
        success=True,
        message="登录成功",
        data=TokenResponse(
            access_token=token,
            username=user["username"],
            role=user.get("role", "user"),
        ),
    )


@router.get("/me", response_model=ApiResponse, summary="获取当前用户信息")
def me(current_user: dict = Depends(get_current_user)):
    """验证 Token 有效性，返回当前用户详细信息。"""
    user = get_user_by_id(current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user_info = UserInfo(
        id=user["id"],
        username=user["username"],
        role=user["role"],
        email=user.get("email", ""),
        avatar=user.get("avatar", ""),
        status=user.get("status", "active"),
        created_at=str(user["created_at"]) if user.get("created_at") else None,
    )
    return ApiResponse(success=True, data=user_info)
