"""
认证路由 (Phase 4) — HMAC-SHA256 签名 token (无第三方依赖)

端点:
  POST /api/v1/auth/login    用户名密码 → {token, user}
  GET  /api/v1/auth/profile  解析 Bearer → 当前用户

Token 格式: "{user_id}.{role}.{expiry}.{hmac}"
  hmac = HMAC-SHA256(jwt_secret, "user_id.role.expiry")

chat 路由通过 get_current_user() 依赖或 resolve_user_id() 解析当前用户,
用于订单等业务的归属校验 (Phase 2 的 user_id 来源升级)。
"""

import hashlib
import hmac
import logging
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["认证"])


class LoginRequest(BaseModel):
    username: str
    password: str


def _sign(payload: str) -> str:
    return hmac.new(settings.jwt_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_token(user_id: str, role: str = "user", ttl: Optional[int] = None) -> str:
    """签发 token: user_id.role.expiry.signature"""
    exp = int(time.time()) + (ttl or settings.token_ttl)
    payload = f"{user_id}.{role}.{exp}"
    return f"{payload}.{_sign(payload)}"


def decode_token(token: str) -> Optional[dict]:
    """解析 token, 无效/过期返回 None"""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    payload = ".".join(parts[:3])
    sig = parts[3]
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    try:
        user_id, role, exp = payload.split(".")
    except ValueError:
        return None
    if int(exp) < time.time():
        return None
    return {"user_id": user_id, "role": role}


def resolve_user_id(authorization: Optional[str] = None) -> Optional[str]:
    """从 Authorization: Bearer <token> 解析 user_id (无效返回 None)"""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    data = decode_token(token.strip())
    return data["user_id"] if data else None


def require_roles(*roles: str):
    """FastAPI 依赖: 校验 Bearer token 角色 (后端强制鉴权)

    用法: 挂到管理类 router/接口, 非指定角色直接 401/403。
    """
    def checker(authorization: Optional[str] = Header(None)):
        data = decode_token(authorization.partition(" ")[2] if authorization else "")
        if not data:
            raise HTTPException(status_code=401, detail="未认证或 token 无效")
        if data["role"] not in roles:
            raise HTTPException(status_code=403, detail="权限不足，需要角色: " + "/".join(roles))
        return data
    return checker


@router.post("/login")
async def login(req: LoginRequest):
    """用户名密码登录 → 签发 token (双端: admin=全部 / user=仅客服)"""
    # 管理员端
    if settings.admin_password and req.username == settings.admin_username \
            and req.password == settings.admin_password:
        token = create_token(req.username, role="admin")
        logger.info("管理员登录成功: %s", req.username)
        return {"token": token, "user": {"username": req.username, "role": "admin"}}
    # 用户端
    if settings.user_password and req.username == settings.user_username \
            and req.password == settings.user_password:
        token = create_token(req.username, role="user")
        logger.info("用户登录成功: %s", req.username)
        return {"token": token, "user": {"username": req.username, "role": "user"}}
    raise HTTPException(status_code=401, detail="用户名或密码错误")


@router.get("/profile")
async def profile(authorization: Optional[str] = Header(None)):
    """当前登录用户信息"""
    data = decode_token(authorization.partition(" ")[2] if authorization else "")
    if not data:
        raise HTTPException(status_code=401, detail="未认证或 token 无效")
    return {"username": data["user_id"], "role": data["role"]}
