import logging
import os
import sys
import fcntl
import multiprocessing

from fastapi import FastAPI
from uvicorn import run

from weather_radar.lib import cachewriter, hotcache

from . import api, app

logger = logging.getLogger()

# Guards against the uvicorn workers each spawning background processes: only
# the worker that wins this file lock spawns the single cache-writer and
# hot-cache scheduler. The lock fd is kept open for the life of the process
# (releasing it on exit lets a future restart re-elect a spawner).
_LEADER_LOCK_PATH = os.path.join(
    os.environ.get("CACHE_DIR", "/tmp"), "weather-radar-cachewriter.lock"
)
_leader_lock_fd = None
_cache_writer_proc = None
_hotcache_proc = None


def _try_become_writer_spawner():
    """Return True if this process won the right to spawn the writer."""
    global _leader_lock_fd
    try:
        fd = os.open(_LEADER_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return False
    _leader_lock_fd = fd  # keep open to hold the lock
    return True


def _port_open(host, port, timeout=0.25):
    """True if something is already accepting connections at host:port."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_background_processes():
    """Spawn the single cache-writer AND hot-cache scheduler processes (once
    across all workers) and wait until their manager servers are accepting
    connections. Workers that don't win the spawner lock connect to them
    lazily on first use.

    Tolerant of an already-running manager (e.g. a stale process after a
    --reload restart, or a socket not yet released): if the port is already
    open, we skip spawning and let workers connect to the existing server,
    avoiding an 'address already in use' crash.
    """
    global _cache_writer_proc, _hotcache_proc
    if not _try_become_writer_spawner():
        logger.warning("background process spawn skipped (another worker owns it)")
        return False

    # 1. Cache writer (owns disk writes).
    if _port_open(cachewriter.CACHE_WRITER_HOST, cachewriter.CACHE_WRITER_PORT):
        logger.warning("cache writer already listening; reusing it")
    else:
        writer_ready = multiprocessing.Event()
        writer = multiprocessing.Process(
            target=cachewriter.run_writer,
            args=(writer_ready,),
            name="cache-writer",
            daemon=True,
        )
        writer.start()
        writer_ready.wait(timeout=10)
        _cache_writer_proc = writer
        logger.warning("cache writer process started (pid=%s)", writer.pid)

    # 2. Hot-cache scheduler (read-only client of the writer; keeps watched
    #    URLs warm by refreshing paths just before they expire).
    if _port_open(hotcache.SCHED_HOST, hotcache.SCHED_PORT):
        logger.warning("hotcache scheduler already listening; reusing it")
    else:
        sched_ready = multiprocessing.Event()
        sched = multiprocessing.Process(
            target=hotcache.run_scheduler,
            args=(sched_ready,),
            name="hotcache-scheduler",
            daemon=True,
        )
        sched.start()
        sched_ready.wait(timeout=10)
        _hotcache_proc = sched
        logger.warning("hotcache scheduler process started (pid=%s)", sched.pid)
    # This worker won the leader lock (whether or not it had to spawn fresh
    # processes vs. reuse existing ones).
    return True


def server():

    logroot = os.environ.get("LOG_DIR", "/tmp")
    filename = os.path.join(logroot, "weather_radar.log")
    logging.basicConfig(
        filename=filename,
        level=getattr(logging, str(os.environ.get("LOG_LEVEL")), logging.WARNING)
    )

    logger.addHandler(logging.StreamHandler(sys.stdout))

    server = FastAPI()

    _install_access_control(server)

    @server.on_event("startup")
    def _startup():
        # Each uvicorn worker runs this; the file lock ensures exactly one
        # (the "leader") actually spawns the background processes (writer +
        # scheduler). The leader ALSO runs the single-instance bundle builder,
        # which precomputes watched URLs' animation bundles to disk — built
        # once, shared to all workers via the filesystem (no per-worker work).
        is_leader = _start_background_processes()
        if is_leader:
            from .bundle import start_builder
            start_builder()

    @server.get("/health")
    def _health():
        return {"ok": True}

    server.include_router(api.router)
    server.include_router(app.router)

    return server


def _install_access_control(server):
    """Gate every route behind an access token WHEN tokens are configured.

    A request is allowed if it presents a valid token via `?token=`, the
    `X-Access-Token` header, or the `wr_token` cookie (best-effort — iOS
    WebKit ITP often blocks cross-origin iframe cookies, so the token also
    rides the URL and is threaded into client API calls). When NO access
    tokens are configured the app is open (dev default); localhost is also
    always allowed so a local dev run is never locked out.

    EXEMPT paths (never token-gated): /health, and the admin surface
    (/admin, /api/admin/... and the admin login/token endpoints), which carry
    their OWN password-based auth. This lets an operator reach the admin page
    to mint the first token without already having one.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse, RedirectResponse
    from weather_radar.lib import auth

    EXEMPT_PREFIXES = ("/health", "/admin", "/api/admin", "/api/auth",
                       "/not-authorized", "/login")

    @server.middleware("http")
    async def _access_control(request: Request, call_next):
        path = request.url.path
        if _path_exempt(path, EXEMPT_PREFIXES):
            return await call_next(request)

        # Open ONLY when no tokens are configured (fresh/dev install). Once a
        # token exists, EVERY request must present one — no localhost bypass.
        if not auth.access_tokens_configured():
            return await call_next(request)

        supplied = (request.query_params.get("token")
                    or request.headers.get("X-Access-Token")
                    or request.cookies.get("wr_token"))
        if auth.verify_access_token(supplied):
            response = await call_next(request)
            # Best-effort cookie so same-origin navigations don't need the
            # token re-appended (ignored by ITP in cross-origin iframes, which
            # is why the token also threads through the URL/API calls).
            try:
                response.set_cookie("wr_token", supplied, max_age=7 * 24 * 3600,
                                    httponly=True, secure=True, samesite="none")
            except Exception:
                pass
            return response

        # Unauthorized. API / XHR callers get a JSON 401. For a top-level PAGE
        # navigation:
        #   * NO token supplied        -> /login (prompt to authenticate)
        #   * token supplied but WRONG -> /not-authorized (dead-end; a bad/
        #     rotated/revoked token shouldn't re-loop through login, and an
        #     iframe with a stale token gets the clean message)
        accept = request.headers.get("accept", "")
        is_page = (not path.startswith("/api")) and ("text/html" in accept)
        if is_page:
            if supplied:
                return RedirectResponse("/not-authorized", status_code=302)
            from urllib.parse import quote
            nxt = request.url.path
            if request.url.query:
                nxt += "?" + request.url.query
            return RedirectResponse("/login?next=" + quote(nxt, safe=""),
                                    status_code=302)
        return JSONResponse({"error": "unauthorized"}, status_code=401)


def _path_exempt(path, prefixes):
    for p in prefixes:
        if path == p or path.startswith(p + "/"):
            return True
    return False


def main():
    run(server(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
