"""
auth.py — JWT authentication and authorisation helpers for MaintiDoc.

Provides:
- Password hashing / verification via bcrypt (passlib).
- JWT creation / decoding via python-jose.
- FastAPI dependencies: get_current_user, require_role.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.crud import get_user_by_username
from backend.db.database import get_db
from backend.db.models import User

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY: str = settings.app_secret_key
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8  # 8 hours

# OAuth2 bearer scheme — tokenUrl points at the login endpoint
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

# Passlib context for bcrypt hashing
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Return the bcrypt hash of *password*."""
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a signed JWT containing *data* plus an *exp* claim.

    Args:
        data: Payload to embed (e.g. {"sub": username, "role": role}).
        expires_delta: Override the default expiry window.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.now(tz=timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT string.

    Args:
        token: Raw bearer token from the Authorization header.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException 401: if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except ExpiredSignatureError:
        logger.warning("Token has expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as exc:
        logger.warning(f"Invalid token: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    token: str = Depends(_oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency that resolves the current authenticated user.

    Decodes the bearer token, looks up the user in the database, and
    verifies the account is active.

    Args:
        token: JWT bearer token (injected by OAuth2PasswordBearer).
        db: Async database session (injected by get_db).

    Returns:
        The authenticated User ORM instance.

    Raises:
        HTTPException 401: on invalid token or missing user.
        HTTPException 403: if the account is inactive.
    """
    payload = decode_token(token)
    username: str | None = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing 'sub' claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_user_by_username(db, username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account.",
        )

    return user


def require_role(*roles: str):
    """
    Return a FastAPI dependency that checks the current user's role.

    Usage::

        @router.post("/admin", dependencies=[Depends(require_role("admin", "manager"))])
        async def admin_only(): ...

    Args:
        *roles: Allowed role strings (e.g. "admin", "manager").

    Returns:
        Async FastAPI dependency callable.

    Raises:
        HTTPException 403: if the user's role is not in *roles*.
    """

    async def _check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role(s): {list(roles)}. Your role: {current_user.role}",
            )
        return current_user

    return _check_role
