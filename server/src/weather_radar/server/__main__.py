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
        return None

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
    return _cache_writer_proc


def server():

    logroot = os.environ.get("LOG_DIR", "/tmp")
    filename = os.path.join(logroot, "weather_radar.log")
    logging.basicConfig(
        filename=filename,
        level=getattr(logging, str(os.environ.get("LOG_LEVEL")), logging.WARNING)
    )

    logger.addHandler(logging.StreamHandler(sys.stdout))

    server = FastAPI()

    @server.on_event("startup")
    def _startup():
        # Each uvicorn worker runs this; the file lock ensures exactly one
        # actually spawns the background processes (writer + scheduler).
        _start_background_processes()
        # Every worker starts its OWN spline warmer: the fitted splines live in
        # per-worker memory, so each worker must warm its own _MODEL_CACHE for
        # the watched URLs (unlike the shared on-disk NOAA cache).
        from .warmer import start as _start_warmer
        _start_warmer()

    server.include_router(api.router)
    server.include_router(app.router)

    return server


def main():
    run(server(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
