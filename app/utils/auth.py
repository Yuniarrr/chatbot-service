import logging
import jwt

from uuid import UUID
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Annotated, Any
from pydantic import BaseModel

from app.env import (
    JWT_SECRET_KEY,
    API_KEY,
    PY_ENV,
    JWT_ACCESS_TOKEN_EXPIRE,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from app.core.constants import ERROR_MESSAGES
from app.models.users import users, Role
from app.core.database import async_get_db

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

bearer_security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

log = logging.getLogger(__name__)
log.setLevel("UTILS")


class TokenData(BaseModel):
    email: Optional[str] = None
    id: Optional[str] = None
    role: Role = Role.USER


def verify_password(plain_password, hashed_password):
    return (
        pwd_context.verify(plain_password, hashed_password) if hashed_password else None
    )


def get_password_hash(password):
    return pwd_context.hash(password)


async def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = {
        key: str(value) if isinstance(value, UUID) else value
        for key, value in data.items()
    }

    if expires_delta:
        expire = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None) + expires_delta
    else:
        expire = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None) + timedelta(
            hours=JWT_ACCESS_TOKEN_EXPIRE
        )
    to_encode.update({"exp": expire})
    encoded_jwt: str = jwt.encode(to_encode, JWT_SECRET_KEY)
    return encoded_jwt


async def create_refresh_token(
    data: dict[str, Any], expires_delta: Union[timedelta, None] = None
) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None) + expires_delta
    else:
        expire = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None) + timedelta(
            days=REFRESH_TOKEN_EXPIRE_DAYS
        )
    to_encode.update({"exp": expire})
    encoded_jwt: str = jwt.encode(to_encode, JWT_SECRET_KEY)
    return encoded_jwt


def decode_token(token: str) -> Union[TokenData, None]:
    try:
        decoded = jwt.decode(token, JWT_SECRET_KEY)

        email: Optional[str] = decoded.get("email")
        user_id: Optional[str] = decoded.get("id")
        role: Optional[str] = decoded.get("role")

        return TokenData(email=email, id=user_id, role=role)
    except Exception:
        return None


def extract_token_from_auth_header(auth_header: str):
    return auth_header[len("Bearer ") :]


def get_http_authorization_cred(auth_header: str):
    try:
        scheme, credentials = auth_header.split(" ")
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)
    except Exception:
        raise ValueError(ERROR_MESSAGES.INVALID_TOKEN_OR_API_KEY("Bearer token"))


def get_current_user(
    db: Annotated[AsyncSession, Depends(async_get_db)],
    request: Request,
    auth_token: HTTPAuthorizationCredentials = Depends(bearer_security),
):
    token = None
    api_key = request.headers.get("x-api-key")

    if auth_token is not None:
        token = auth_token.credentials

    if token is None:
        raise HTTPException(
            status_code=403,
            detail=ERROR_MESSAGES.MISSING_TOKEN_OR_API_KEY("Bearer token"),
        )

    if PY_ENV != "dev":
        if api_key is None:
            raise HTTPException(
                status_code=403,
                detail=ERROR_MESSAGES.MISSING_TOKEN_OR_API_KEY("x-api-key"),
            )

        if api_key != API_KEY:
            raise HTTPException(
                status_code=403,
                detail=ERROR_MESSAGES.INVALID_TOKEN_OR_API_KEY("x-api-key"),
            )

    try:
        data = decode_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN_OR_API_KEY("Bearer token"),
        )

    if data is not None and "email" in data:
        user = users.get(db, email=data["email"])
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.INVALID_TOKEN_OR_API_KEY("Bearer token"),
            )

        return user
    elif data is not None and "id" in data:
        user = users.get(db, email=data["id"])
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.INVALID_TOKEN_OR_API_KEY("Bearer token"),
            )

        return user
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )


def get_verified_user(user=Depends(get_current_user)):
    if user.role not in {
        Role.USER.value,
        Role.ADMINISTRATOR.value,
        Role.DEVELOPER.value,
    }:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    return user


def get_admin_user(user=Depends(get_current_user)):
    if user.role != Role.ADMINISTRATOR.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    return user


def get_developer_user(user=Depends(get_current_user)):
    if user.role != Role.DEVELOPER.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    return user


def get_not_user(user=Depends(get_current_user)):
    if user.role == Role.USER.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    return user
