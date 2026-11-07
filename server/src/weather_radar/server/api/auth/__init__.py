"""Admin auth + access-token management API.

Password-gated (NOT token-gated — this is how an operator mints the first
token). Flow:
  POST /api/auth/login   {password}        -> sets a signed admin session cookie
  POST /api/auth/logout                    -> clears it
  GET  /api/auth/tokens                    -> list token metadata (no secrets)
  POST /api/auth/tokens  {label?, ttl?}    -> generate a token (secret shown ONCE)
  DELETE /api/auth/tokens/{id}             -> revoke a token

Authorization for the token endpoints: a valid admin session cookie, OR —
when NO admin password is configured — localhost only (safe dev default,
mirroring the hot-cache admin).
"""

import logging

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from weather_radar.lib import auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth")

_ADMIN_COOKIE = "wr_admin"


def _require_admin(request: Request):
    """Authorize an admin action: valid session cookie, or (no password set)
    localhost. Raises 401/403 otherwise."""
    session = request.cookies.get(_ADMIN_COOKIE)
    if session and auth.verify_admin_session(session):
        return
    if not auth.admin_password_configured():
        client = request.client.host if request.client else None
        if auth.is_localhost(client):
            return
        raise HTTPException(status_code=403,
                            detail="admin is localhost-only until WR_ADMIN_PASSWORD[_HASH] is set")
    raise HTTPException(status_code=401, detail="admin login required")


class LoginBody(BaseModel):
    password: str


@router.post("/login", response_class=JSONResponse)
def login(request: Request, body: LoginBody):
    """Verify the admin password; on success set a signed session cookie."""
    if not auth.admin_password_configured():
        # No password configured: only localhost may "log in" (and it doesn't
        # strictly need to — the token endpoints already allow localhost).
        client = request.client.host if request.client else None
        if not auth.is_localhost(client):
            raise HTTPException(status_code=403, detail="admin password not configured")
        resp = JSONResponse({"ok": True, "note": "no password configured; localhost access"})
        return resp
    if not auth.verify_admin_password(body.password):
        raise HTTPException(status_code=401, detail="invalid password")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(_ADMIN_COOKIE, auth.mint_admin_session(),
                    httponly=True, secure=True, samesite="strict",
                    max_age=12 * 3600)
    return resp


@router.post("/logout", response_class=JSONResponse)
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(_ADMIN_COOKIE)
    return resp


@router.get("/status", response_class=JSONResponse)
def status(request: Request):
    """Whether the current caller is an authenticated admin, and whether a
    password is even configured (so the UI can adapt)."""
    session = request.cookies.get(_ADMIN_COOKIE)
    authed = bool(session and auth.verify_admin_session(session))
    if not authed and not auth.admin_password_configured():
        client = request.client.host if request.client else None
        authed = auth.is_localhost(client)
    return {"authenticated": authed,
            "password_configured": auth.admin_password_configured()}


class LoginTokenBody(BaseModel):
    password: str = ""
    token: str = ""


@router.post("/login-token", response_class=JSONResponse)
def login_token(request: Request, body: LoginTokenBody):
    """Used by the /login page to authenticate. Login never mints/rotates — it
    hands back an EXISTING access token (and, for the password path, also
    establishes the admin session so the admin pages work).

    Two ways:
      * password -> verify the admin password. Sets the admin session cookie
        AND returns the newest existing access token (or token=null if none
        exist yet — fine when heading to an admin page, which needs only the
        session).
      * token    -> if the pasted token is itself valid, echo it back (this
        path does NOT grant admin — a token is app access, not admin rights).

    Returns {ok, token|null, admin: bool}. The caller redirects to `next`
    (appending ?token= when a token is present and the destination needs it).
    """
    pasted = (body.token or "").strip()
    if pasted:
        if auth.verify_access_token(pasted):
            return {"ok": True, "token": pasted, "admin": False}
        raise HTTPException(status_code=401, detail="invalid token")

    password = body.password or ""
    if not password:
        raise HTTPException(status_code=400, detail="password or token required")
    if not auth.admin_password_configured():
        raise HTTPException(status_code=403, detail="admin password not configured")
    if not auth.verify_admin_password(password):
        raise HTTPException(status_code=401, detail="invalid password")
    tok = auth.newest_token()   # may be None (no tokens yet)
    resp = JSONResponse({"ok": True, "token": tok, "admin": True})
    # Password login also grants admin (session cookie) so the admin pages are
    # reachable right after signing in.
    resp.set_cookie(_ADMIN_COOKIE, auth.mint_admin_session(),
                    httponly=True, secure=True, samesite="strict",
                    max_age=12 * 3600)
    return resp


@router.get("/tokens", response_class=JSONResponse)
def tokens_list(request: Request):
    _require_admin(request)
    return {"tokens": auth.list_tokens()}
class TokenBody(BaseModel):
    label: str = ""
    ttl_seconds: float | None = None


@router.post("/tokens", response_class=JSONResponse)
def tokens_create(request: Request, body: TokenBody):
    _require_admin(request)
    created = auth.generate_token(label=body.label, ttl_seconds=body.ttl_seconds)
    # The full secret is returned ONCE here for the operator to copy.
    return {"ok": True, "created": created}


@router.delete("/tokens/{token_id}", response_class=JSONResponse)
def tokens_revoke(request: Request, token_id: str):
    _require_admin(request)
    removed = auth.revoke_token(token_id)
    if not removed:
        raise HTTPException(status_code=404, detail="token not found")
    return {"ok": True}
