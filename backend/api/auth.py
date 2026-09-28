"""
api/auth.py — Enterprise JWT Auth + RBAC + API Key Management
──────────────────────────────────────────────────────────────
Features:
  • JWT Bearer token auth (HS256, access + refresh tokens)
  • API Key auth (X-API-Key header) as alternative
  • Role-Based Access Control: admin | user | readonly
  • Dependency injection helpers for FastAPI routes
  • Audit logging (every auth event logged with user/IP/action)
  • Timing-safe comparisons (prevent timing attacks)
  • Graceful bypass when AUTH_ENABLED=false (dev mode)
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel

from src.config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Roles
# ─────────────────────────────────────────────────────────────────────────────

class Role(str, Enum):
    ADMIN    = "admin"     # full access: upload, chat, eval, config
    USER     = "user"      # upload + chat
    READONLY = "readonly"  # chat only, no upload


ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN:    {"chat", "upload", "evaluate", "admin", "delete"},
    Role.USER:     {"chat", "upload"},
    Role.READONLY: {"chat"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Token models
# ─────────────────────────────────────────────────────────────────────────────

class TokenPayload(BaseModel):
    sub: str                # user_id
    role: Role = Role.USER
    jti: str = ""           # JWT ID (for revocation)
    exp: Optional[int] = None


class AuthUser(BaseModel):
    user_id: str
    role: Role
    token_type: str = "bearer"   # bearer | api_key


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60


# ─────────────────────────────────────────────────────────────────────────────
# Token creation
# ─────────────────────────────────────────────────────────────────────────────

def _create_token(
    user_id: str,
    role: Role,
    expires_delta: timedelta,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  user_id,
        "role": role.value,
        "jti":  str(uuid.uuid4()),
        "iat":  now,
        "exp":  now + expires_delta,
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)


def create_access_token(user_id: str, role: Role = Role.USER) -> str:
    return _create_token(
        user_id, role,
        timedelta(minutes=Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str, role: Role = Role.USER) -> str:
    return _create_token(
        user_id, role,
        timedelta(days=Config.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
    )


def create_token_pair(user_id: str, role: Role = Role.USER) -> TokenPair:
    return TokenPair(
        access_token=create_access_token(user_id, role),
        refresh_token=create_refresh_token(user_id, role),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Token decoding
# ─────────────────────────────────────────────────────────────────────────────

def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(
            token,
            Config.JWT_SECRET_KEY,
            algorithms=[Config.JWT_ALGORITHM],
        )
        return TokenPayload(
            sub=payload["sub"],
            role=Role(payload.get("role", "user")),
            jti=payload.get("jti", ""),
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# API Key validation (timing-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison (prevents timing attacks)."""
    return hmac.compare_digest(
        hashlib.sha256(a.encode()).digest(),
        hashlib.sha256(b.encode()).digest(),
    )


def validate_api_key(api_key: str) -> Optional[AuthUser]:
    """
    Validate an API key. In production, look up from database.
    For now: checks against ADMIN_API_KEY from config.
    Returns AuthUser or None.
    """
    if not Config.ADMIN_API_KEY:
        return None
    if _safe_compare(api_key, Config.ADMIN_API_KEY):
        return AuthUser(user_id="api-key-admin", role=Role.ADMIN, token_type="api_key")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI security schemes
# ─────────────────────────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_scheme = APIKeyHeader(name=Config.API_KEY_HEADER, auto_error=False)


# ─────────────────────────────────────────────────────────────────────────────
# Audit logging
# ─────────────────────────────────────────────────────────────────────────────

def _audit(
    request: Request,
    user: Optional[AuthUser],
    action: str,
    success: bool,
    detail: str = "",
) -> None:
    client_ip = request.client.host if request.client else "unknown"
    user_id = user.user_id if user else "anonymous"
    logger.info(
        f"[AUDIT] action={action} user={user_id} role={user.role.value if user else 'none'} "
        f"ip={client_ip} path={request.url.path} success={success} detail={detail}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core dependency: get_current_user
# ─────────────────────────────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    api_key: Optional[str] = Security(_api_key_scheme),
) -> AuthUser:
    """
    FastAPI dependency that returns the authenticated user.
    Tries Bearer JWT first, then API key.
    If AUTH_ENABLED=false, returns a default admin user (dev mode).
    """
    # Dev mode bypass
    if not Config.AUTH_ENABLED:
        return AuthUser(user_id="dev-user", role=Role.ADMIN)

    # Try Bearer JWT
    if bearer and bearer.credentials:
        try:
            payload = decode_token(bearer.credentials)
            user = AuthUser(user_id=payload.sub, role=payload.role)
            _audit(request, user, "authenticate", True, "jwt")
            return user
        except HTTPException:
            pass   # fall through to API key

    # Try API Key
    if api_key:
        user = validate_api_key(api_key)
        if user:
            _audit(request, user, "authenticate", True, "api_key")
            return user

    _audit(request, None, "authenticate", False, "no valid credentials")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide a Bearer token or API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Permission checkers (RBAC)
# ─────────────────────────────────────────────────────────────────────────────

def require_permission(permission: str):
    """Dependency factory: raises 403 if user lacks permission."""
    async def _check(
        request: Request,
        user: AuthUser = Depends(get_current_user),
    ) -> AuthUser:
        allowed = ROLE_PERMISSIONS.get(user.role, set())
        if permission not in allowed:
            _audit(request, user, f"permission:{permission}", False, "forbidden")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required. Your role: {user.role.value}",
            )
        _audit(request, user, f"permission:{permission}", True)
        return user
    return _check


def require_admin():
    """Shortcut: require admin role."""
    return require_permission("admin")


# Convenient pre-built dependencies
CanChat   = Depends(require_permission("chat"))
CanUpload = Depends(require_permission("upload"))
CanEval   = Depends(require_permission("evaluate"))
IsAdmin   = Depends(require_admin())
