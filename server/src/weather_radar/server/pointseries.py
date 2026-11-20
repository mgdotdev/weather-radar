"""Cached single-point forecast series (graph / forecast pages) and the
background PRE-WARM of the ones a watched URL implies.

The graph endpoint (/api/models/{kind}/params/{attribute}) answers from a disk
cache (weather_radar.lib.seriescache) when it can, so a request costs a file
read instead of re-parsing ~100 full-grid frames in a fresh worker. A miss
computes the series as before and saves it for the next request (any worker).

PRE-WARM — the bundle builder calls warm_url() for every watched URL, which
keeps these series ready for exactly:
  * a watched MAP url:  the points on its map (its ``pins``), for the map's
    current layer only — what a pin's "graph" / "forecast" buttons open;
  * a watched GRAPH url: the explicit (models, lat, lon, dt) it names.
Nothing else is warmed; any other point/attribute is computed on first request
and cached from then on.
"""

import json
import logging
import threading
from urllib.parse import urlparse, parse_qs

from weather_radar.lib import seriescache

logger = logging.getLogger(__name__)

# The steps a pin's buttons open: the graph page's default dt, and the forecast
# page (which sends no dt, i.e. the hourly default).
GRAPH_DEFAULT_DT = 450
PIN_STEPS = (GRAPH_DEFAULT_DT, 3600)


def _grib_source():
    from weather_radar.lib.models import weather_source
    return weather_source() in ("ndfd", "nbm")


def _token(kind, param):
    """Current data-version token for this attribute (None if unknown)."""
    try:
        from weather_radar.server.bundle import _source_token
        return _source_token(param, kind)
    except Exception:
        return None


def _compute(kind, param, lat, lon, dt):
    from weather_radar.server.api import models as api
    return api._point_series(kind, param, lat, lon, dt)


def _store(path, data, token, managed):
    """Save through the cache-writer; write inline only if it's unreachable."""
    blob = seriescache.pack(data, token, managed)
    try:
        from weather_radar.lib import cachewriter
        return cachewriter.write_series_cache(path, blob, token, managed)
    except Exception:
        return seriescache.save(path, blob, token, managed)


def _store_in_background(path, data, token, managed=False):
    def go():
        try:
            _store(path, data, token, managed)
        except Exception:
            logger.warning("series cache save failed", exc_info=True)
    threading.Thread(target=go, name="series-cache-save", daemon=True).start()


def point_series(kind, param, lat, lon, dt):
    """The series for the graph endpoint: from the cache when current, else
    computed (and cached for next time). Gridded sources only (NBM/NDFD and the
    AQI grid) — the JSON gridpoints source has no cheap data-version check, so
    it never reaches here."""
    token = _token(kind, param)
    path = seriescache.path_for(kind, param, lat, lon, dt)
    if token is not None:
        hit = seriescache.get(path, token)
        if hit is not None:
            return hit
    result = _compute(kind, param, lat, lon, dt)
    if result and result.get("features"):
        # Read the token AFTER computing: it labels the data actually used (and
        # exists even on the first request, before the source was on disk).
        tok = _token(kind, param)
        if tok is not None:
            _store_in_background(path, result, tok)
    return result


# --- Pre-warm ---------------------------------------------------------------

def targets_for_url(url):
    """[(kind, attribute, lat, lon, dt)] a watched URL asks to keep warm."""
    out = []
    try:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        q = parse_qs(parsed.query)
    except Exception:
        return out

    def num(name):
        try:
            return float(q[name][0])
        except (KeyError, IndexError, ValueError):
            return None

    if len(parts) >= 3 and parts[0] == "map" and parts[1] == "models":
        from weather_radar.server.app import _profile
        profile = _profile(parts[2])
        if profile is None:
            return out
        kind, attr = profile["kind"], profile["attribute"]
        for item in (q.get("pins", [""])[0]).split(";"):
            f = item.split(",")
            try:
                lat, lon = float(f[0]), float(f[1])
            except (ValueError, IndexError):
                continue
            for dt in PIN_STEPS:
                out.append((kind, attr, lat, lon, dt))
    elif parts == ["graph"]:
        lat, lon = num("lat"), num("lon")
        try:
            models = json.loads(q.get("models", ["{}"])[0])
        except ValueError:
            models = {}
        dt = int(num("dt") or GRAPH_DEFAULT_DT)
        if lat is not None and lon is not None and isinstance(models, dict):
            for kind, attrs in models.items():
                for attr in (attrs if isinstance(attrs, list) else []):
                    out.append((str(kind), str(attr), lat, lon, dt))

    seen, uniq = set(), []
    for t in out:
        key = (t[0], t[1], round(t[2], 6), round(t[3], 6), seriescache.normalize_step(t[4]))
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def warm(kind, param, lat, lon, dt):
    """Make sure one series is cached for the current data, marked as managed
    (so it keeps being served while the next cycle's is built). Returns one of
    "fresh", "marked", "built", "skipped", "empty"."""
    if kind != "aqi" and not _grib_source():
        return "skipped"      # (AQI is gridded whatever the weather source is)
    token = _token(kind, param)
    if token is None:
        return "skipped"
    path = seriescache.path_for(kind, param, lat, lon, dt)
    if seriescache.is_fresh(path, token):
        entry = seriescache.read(path)
        if entry.get("managed"):
            return "fresh"
        # A request cached it first: same data, just promote it to managed.
        _store(path, entry["data"], entry["token"], True)
        return "marked"
    result = _compute(kind, param, lat, lon, dt)
    if not result or not result.get("features"):
        return "empty"
    _store(path, result, _token(kind, param) or token, True)
    return "built"


def warm_url(url):
    """Pre-warm everything a watched URL implies. Returns {status: count}."""
    counts = {}
    for kind, attr, lat, lon, dt in targets_for_url(url):
        try:
            status = warm(kind, attr, lat, lon, dt)
        except Exception:
            logger.warning("series pre-warm failed: %s %s %.4f,%.4f dt=%s",
                           kind, attr, lat, lon, dt, exc_info=True)
            status = "error"
        counts[status] = counts.get(status, 0) + 1
    if counts.get("built"):
        logger.warning("series pre-warm %s: %s", url.split("?")[0], counts)
    return counts
