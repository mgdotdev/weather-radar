"""Per-worker spline warmer.

Keeps the *fitted* per-gridpoint splines hot in THIS worker's process-local
``_MODEL_CACHE`` for the admin "watched" URLs, so a user hitting a watched URL
finds every gridpoint model already fitted (``models_cold == 0``) instead of
paying the fit cost on the request path.

Why per-worker
--------------
The hot-cache *scheduler* (``lib/hotcache.py``) keeps the NOAA **cache files**
(``/points``, ``/gridpoints``) warm on disk — shared across all processes via
the filesystem. But the fitted splines live in each worker's own memory
(``models.observation._MODEL_CACHE`` / ``models.accumulation._MODEL_CACHE``),
which is NOT shared between processes. The scheduler cannot populate them.

So each worker runs its own warmer thread. It is a *read-only client* of the
scheduler: it asks ``hotcache.list_watched()`` for the authoritative URL list
(single source of truth, persisted + admin-controlled) and fits those URLs'
models into this worker's cache. With N workers this costs N× the fit CPU and
N× the spline memory — acceptable for a modest watch-list and a small worker
count (we run 4).

Design
------
* One daemon thread per worker, started from the FastAPI ``startup`` hook.
* Gated by ``WR_WARM_SPLINES`` (default ON); set to ``0`` to disable.
* Loop: pull the watched URLs, warm each, then sleep ``WARM_INTERVAL_SEC``.
* Warming a URL = resolve its gridpoints (already cheap: the scheduler keeps
  ``/points`` warm) and touch ``ensemble[gp].model`` for each gridpoint. The
  model cache refits only when the fit is missing or its source data has
  expired (``is_expired``), so steady-state passes are nearly free; a pass
  right after the scheduler refreshes a file does the (cheap, local) refit
  BEFORE a user request would have to.
* Best-effort throughout: any failure (scheduler down, one bad gridpoint, a
  transiently missing file) is logged and skipped; the warmer never raises
  into the worker.
"""

import os
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Gate: default on. Set WR_WARM_SPLINES=0 to turn the warmer off entirely.
ENABLED = os.environ.get("WR_WARM_SPLINES", "1") != "0"

# How often each worker re-warms the watched URLs. The refit itself is gated by
# model expiry, so a short interval mostly just re-checks freshness cheaply.
WARM_INTERVAL_SEC = float(os.environ.get("WR_WARM_INTERVAL_SEC", "60"))

# Delay the first pass slightly so worker startup (and the scheduler coming up)
# settles first.
WARM_STARTUP_DELAY_SEC = float(os.environ.get("WR_WARM_STARTUP_DELAY_SEC", "10"))

# Threads used to fit a URL's gridpoint splines in parallel. scipy's spline fit
# releases the GIL during the C routine, so threading (not processes) scales
# the fit without the per-worker _MODEL_CACHE having to be shared. Kept modest
# so 4 workers warming at once don't oversubscribe the host.
WARM_FIT_WORKERS = int(os.environ.get("WR_WARM_FIT_WORKERS", "8"))

_thread = None
_thread_lock = threading.Lock()


def _warm_url(url):
    """Fit (into this worker's model cache) every gridpoint spline for one
    watched map URL. Returns (n_gridpoints, n_fitted_ok) for logging."""
    from urllib.parse import urlparse, parse_qs

    # Imported here (not at module load) so this module stays importable even
    # where the app package isn't — and to avoid pulling scipy-adjacent code
    # into processes that never warm.
    from weather_radar.server.app import _profile, DEFAULT_PROFILE
    from weather_radar.lib.area import MapCoordinate, CoordinateArea
    from weather_radar.server.api.models import ENSEMBLE_KINDS
    from weather_radar.lib.models.accumulation import AccumulationEnsemble
    from weather_radar.lib.models.utils import BoundsError

    parsed = urlparse(url)
    # Only map model URLs carry a contourable layer: /map/models/{model}/
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "map" or parts[1] != "models":
        return (0, 0)
    model = parts[2]

    q = parse_qs(parsed.query)

    def _num(name):
        vals = q.get(name)
        if not vals:
            return None
        try:
            return float(vals[0])
        except (TypeError, ValueError):
            return None

    lat, lon = _num("lat"), _num("lon")
    if lat is None or lon is None:
        return (0, 0)
    width = _num("width") or 1.0
    height = _num("height") or 1.0

    # Resolve the model's ensemble kind + NOAA attribute exactly as the map
    # route does, so we fit the SAME splines a real request would use.
    profile = _profile(model) or {**DEFAULT_PROFILE, "attribute": model}
    attribute = profile["attribute"]
    kind = profile["kind"]
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)

    area = CoordinateArea(MapCoordinate(lat, lon), width, height)
    ensemble = ensemble_class(area, attribute)

    gridpoints = list(area.iter_gridpoints())

    # Fit each gridpoint's spline, in parallel. get_*_model refits only when
    # missing/expired, so a steady-state pass is just cache hits (cheap) and
    # only a cold/first pass or a post-refresh pass does real fitting work.
    # scipy releases the GIL during the fit, so threads give real speedup.
    def _fit_one(gp):
        try:
            ensemble[gp].model  # force the (cached) fit now
            return True
        except BoundsError:
            # Out of coverage / horizon for this gridpoint — nothing to fit.
            return False
        except Exception:
            # One bad gridpoint must not abort the rest of the area.
            logger.debug("warmer: skipping gridpoint %s for %s", gp, attribute,
                         exc_info=True)
            return False

    ok = 0
    if gridpoints:
        workers = max(1, min(WARM_FIT_WORKERS, len(gridpoints)))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="warm-fit") as pool:
            ok = sum(1 for r in pool.map(_fit_one, gridpoints) if r)
    return (len(gridpoints), ok)


def _warm_pass():
    """One full pass: warm every currently-watched URL into this worker."""
    from weather_radar.lib import hotcache

    try:
        watched = hotcache.list_watched()
    except hotcache.SchedulerUnavailable:
        logger.debug("warmer: scheduler unavailable; skipping pass")
        return
    except Exception:
        logger.warning("warmer: failed to list watched URLs", exc_info=True)
        return

    urls = watched.get("urls", []) if isinstance(watched, dict) else []
    if not urls:
        return

    t0 = time.perf_counter()
    total_gp = 0
    total_ok = 0
    for url in urls:
        try:
            n_gp, n_ok = _warm_url(url)
            total_gp += n_gp
            total_ok += n_ok
        except Exception:
            logger.warning("warmer: failed to warm %s", url, exc_info=True)
    dt = time.perf_counter() - t0
    logger.warning(
        "warmer pass: %d url(s) %d gridpoint(s) %d fitted in %.1fs (pid=%s)",
        len(urls), total_gp, total_ok, dt, os.getpid(),
    )


def _run():
    if WARM_STARTUP_DELAY_SEC > 0:
        time.sleep(WARM_STARTUP_DELAY_SEC)
    while True:
        try:
            _warm_pass()
        except Exception:
            # Defensive: the loop must never die on an unexpected error.
            logger.warning("warmer: pass raised; continuing", exc_info=True)
        time.sleep(max(1.0, WARM_INTERVAL_SEC))


def start():
    """Start this worker's warmer thread (idempotent, no-op if disabled)."""
    global _thread
    if not ENABLED:
        logger.warning("spline warmer disabled (WR_WARM_SPLINES=0)")
        return
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        _thread = threading.Thread(target=_run, name="spline-warmer", daemon=True)
        _thread.start()
    logger.warning(
        "spline warmer started (pid=%s interval=%.0fs)",
        os.getpid(), WARM_INTERVAL_SEC,
    )
