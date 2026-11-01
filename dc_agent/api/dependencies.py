from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from ..bootstrap import Bundle
from ..core.errors import AuthError
from ..security import UserContext, authenticate


def get_bundle(request: Request) -> Bundle:
    return request.app.state.bundle


def current_user(authorization: str = Header(default=""), b: Bundle = Depends(get_bundle)) -> UserContext:
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return authenticate(token, b.settings, b.dev_users)
    except AuthError:
        raise HTTPException(status_code=401, detail="unauthorised")
