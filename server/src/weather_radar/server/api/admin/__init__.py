"""Admin API for the hot-cache watch-list.

Lets an operator designate app URLs to be kept "hot" — the hot-cache scheduler
(hotcache.py) proactively refreshes their NOAA data just before it expires, so
navigating to a watched URL finds everything already cached and current.

Access control: these endpoints mutate server state, so they require the
shared admin SESSION (established by the admin password login — see
weather_radar.lib.auth / /api/auth/login). When no admin password is
configured, access falls back to localhost-only (so a bare dev run is usable
but a public deployment isn't wide open by default).
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from weather_radar.lib import hotcache
from weather_radar.lib import auth


router = APIRouter(prefix="/admin")

_LOCALHOSTS = {"127.0.0.1", "::1", "localhost"}


def _authorize(request: Request):
    """Raise 401/403 unless the caller is an authenticated admin.

    Admin auth is the shared admin SESSION (set by the admin password login).
    Only when NO admin password is configured does it fall back to
    localhost-only (bootstrapping). No localhost bypass once a password is set.
    """
    session = request.cookies.get("wr_admin")
    if session and auth.verify_admin_session(session):
        return
    if auth.admin_password_configured():
        raise HTTPException(status_code=401, detail="admin login required")
    # Nothing configured: localhost-only bootstrapping.
    client_host = request.client.host if request.client else None
    if client_host not in _LOCALHOSTS:
        raise HTTPException(
            status_code=403,
            detail="admin is localhost-only until WR_ADMIN_PASSWORD[_HASH] is set",
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


@router.get("/hotcache/stream")
def hotcache_stream(request: Request):
    """Server-Sent Events feed of the watch-list, so the admin page gets PUSH
    updates instead of polling. One long-lived connection per client; the
    server polls the scheduler locally and emits a `data:` event only when the
    serialized list CHANGES, plus a periodic keepalive comment to hold the
    connection through proxies. Authorized by the admin session (EventSource
    sends the same-origin cookie automatically).
    """
    _authorize(request)

    import json as _json
    import time as _time

    def _snapshot():
        try:
            return hotcache.list_watched()
        except hotcache.SchedulerUnavailable:
            return {"error": "scheduler unavailable"}
        except Exception:
            return {"error": "unavailable"}

    def generate():
        # Tell EventSource to wait 3s before reconnecting if the stream drops.
        yield "retry: 3000\n\n"
        last = None
        last_keepalive = _time.time()
        # Poll the (local, cheap) scheduler; push only on change. A keepalive
        # comment every ~15s keeps intermediaries from closing an idle stream.
        while True:
            snap = _snapshot()
            payload = _json.dumps(snap, sort_keys=True)
            now = _time.time()
            if payload != last:
                last = payload
                yield "event: watchlist\ndata: " + payload + "\n\n"
                last_keepalive = now
            elif now - last_keepalive >= 15:
                yield ": keepalive\n\n"
                last_keepalive = now
            _time.sleep(2)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # disable proxy buffering (nginx)
        "Connection": "keep-alive",
    }
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers=headers)


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
