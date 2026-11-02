"""Admin API for the hot-cache watch-list.

Lets an operator designate app URLs to be kept "hot" — the hot-cache scheduler
(hotcache.py) proactively refreshes their NOAA data just before it expires, so
navigating to a watched URL finds everything already cached and current.

Access control: these endpoints mutate server state, so they are gated. If the
ADMIN_TOKEN env var is set, requests must present it (X-Admin-Token header or
?token=). If ADMIN_TOKEN is unset, access is restricted to localhost (so a
bare dev run is usable but a public deployment isn't wide open by default).
"""

import os

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from weather_radar.lib import hotcache


router = APIRouter(prefix="/admin")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
_LOCALHOSTS = {"127.0.0.1", "::1", "localhost"}


def _authorize(request: Request):
    """Raise 401/403 unless the caller is allowed. With ADMIN_TOKEN set, a
    matching token is required; without it, only localhost is allowed."""
    if ADMIN_TOKEN:
        supplied = request.headers.get("X-Admin-Token") or request.query_params.get("token")
        if supplied != ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="invalid or missing admin token")
        return
    client_host = request.client.host if request.client else None
    if client_host not in _LOCALHOSTS:
        raise HTTPException(
            status_code=403,
            detail="admin is localhost-only unless ADMIN_TOKEN is configured",
        )


class WatchBody(BaseModel):
    url: str


@router.get("/hotcache", response_class=JSONResponse)
def hotcache_list(request: Request):
    """List the currently watched URLs and scheduler stats."""
    _authorize(request)
    try:
        return hotcache.list_watched()
    except hotcache.SchedulerUnavailable:
        raise HTTPException(status_code=503, detail="hot-cache scheduler unavailable")


@router.post("/hotcache", response_class=JSONResponse)
def hotcache_add(request: Request, body: WatchBody):
    """Start keeping a URL hot. Returns {ok, paths} on success."""
    _authorize(request)
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        result = hotcache.watch(url)
    except hotcache.SchedulerUnavailable:
        raise HTTPException(status_code=503, detail="hot-cache scheduler unavailable")
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "could not watch url"))
    return result


@router.delete("/hotcache", response_class=JSONResponse)
def hotcache_remove(request: Request, url: str):
    """Stop keeping a URL hot (passed as ?url=)."""
    _authorize(request)
    try:
        return hotcache.unwatch(url)
    except hotcache.SchedulerUnavailable:
        raise HTTPException(status_code=503, detail="hot-cache scheduler unavailable")
