"""Server-side geocoding proxy.

The map's "Drop pin by address…" feature turns a free-form street address into
a lat/lon. Rather than have the browser call the public OpenStreetMap Nominatim
service directly (which leaks the client's IP/UA to a third party, can't be
cached across clients, and makes rate-limit compliance the browser's problem),
the browser calls THIS endpoint and the server proxies to Nominatim.

Centralizing it here lets us:
  * send a descriptive, policy-compliant User-Agent (Nominatim requires one);
  * cache results in-process so repeated lookups don't re-hit the upstream;
  * normalize the response to the single {lat, lon, label} the client needs;
  * swap the upstream provider later without touching the client.

Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/)
asks for a valid identifying User-Agent and <=1 request/second. Geocoding is
driven by an explicit user action (an address prompt), so volume is low; the
cache further reduces upstream calls.
"""

import logging
import os
import threading
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geocode")

# Upstream provider.
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


# Identifying User-Agent. Nominatim's usage policy requires a valid UA that
# identifies the application and carries a contact. Built from env so a
# self-hosted deployment supplies its OWN contact (shared with the NOAA UA):
#
#   GEOCODE_USER_AGENT  full override — if set, used verbatim (highest prio).
#   WR_APP_NAME         app identifier (default "weather-radar").
#   WR_CONTACT_EMAIL    contact; appended as "<app> (self-hosted); <email>".
#
# With no contact configured we still send a valid UA but log a one-time
# warning, since the policy asks for a reachable contact.
_geocode_ua_warned = False


def _user_agent():
    """Return the Nominatim User-Agent. See the note above for precedence."""
    global _geocode_ua_warned
    override = os.environ.get("GEOCODE_USER_AGENT")
    if override and override.strip():
        return override.strip()
    app = (os.environ.get("WR_APP_NAME") or "weather-radar").strip()
    email = (os.environ.get("WR_CONTACT_EMAIL") or "").strip()
    if email:
        return f"{app} (self-hosted); {email}"
    if not _geocode_ua_warned:
        logger.warning(
            "No geocode contact configured: set WR_CONTACT_EMAIL (or "
            "GEOCODE_USER_AGENT) so Nominatim can reach you. Sending a bare "
            "User-Agent %r.", app,
        )
        _geocode_ua_warned = True
    return f"{app} (self-hosted)"

# Upstream request timeout (seconds). A geocode is interactive, so fail fast
# rather than hanging the user's pin drop.
_TIMEOUT = 8

# Small in-process result cache: normalized query -> (expires_epoch, payload).
# Geocoding results are effectively static, so a long TTL is fine; the cap
# bounds memory. Guarded by a lock since uvicorn workers are threaded.
_CACHE_TTL = 24 * 3600
_CACHE_MAX = 512
_cache = {}
_cache_lock = threading.Lock()


# --- Upstream rate limit ----------------------------------------------------
# Nominatim's usage policy caps us at an ABSOLUTE MAXIMUM of 1 request/second.
# This gate serializes the actual upstream calls (cache HITS never reach it) so
# consecutive Nominatim requests are spaced >= _MIN_INTERVAL apart regardless
# of how fast clients type. An interactive geocode can therefore wait briefly
# for a slot; if the backlog would mean waiting longer than _MAX_WAIT, we give
# up with a 503 rather than queueing unboundedly (a pin-drop should fail fast,
# not hang).
#
# CAVEAT: this is PER WORKER PROCESS. uvicorn runs several workers, each with
# its own gate, so the global rate can be up to (workers x 1/s). For a
# self-hosted homelab instance with a handful of users that stays comfortably
# within tolerance; if you run many workers and want a hard global cap, front
# the proxy with a shared limiter (e.g. Redis) or run your own Nominatim.
_MIN_INTERVAL = float(os.environ.get("GEOCODE_MIN_INTERVAL", "1.0"))  # seconds
_MAX_WAIT = float(os.environ.get("GEOCODE_MAX_WAIT", "3.0"))          # seconds


class _RateGate:
    """Serializes upstream calls to at most one per `min_interval` seconds.

    `acquire(max_wait)` blocks until the next slot is available, reserves it,
    and returns True; returns False immediately if the slot is further out than
    `max_wait` (so the caller can shed load instead of queueing). Thread-safe.
    """

    def __init__(self, min_interval):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0  # monotonic epoch of the next permitted call

    def acquire(self, max_wait):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > max_wait:
                return False
            # Reserve this slot (and push the next one out) BEFORE sleeping, so
            # concurrent callers serialize instead of all grabbing "now".
            start = max(now, self._next_allowed)
            self._next_allowed = start + self._min_interval
        if wait > 0:
            time.sleep(wait)
        return True


_rate_gate = _RateGate(_MIN_INTERVAL)


def _cache_get(key):
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        expires, payload = hit
        if expires < now:
            _cache.pop(key, None)
            return None
        return payload


def _cache_put(key, payload):
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            # Evict the soonest-to-expire entry (cheap, bounded, good enough for
            # a tiny interactive cache).
            oldest = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest, None)
        _cache[key] = (time.time() + _CACHE_TTL, payload)


@router.get("/")
def geocode(q: str = "", limit: int = 1, viewbox: str = "", bounded: int = 0):
    """Resolve a free-form address/place `q` to one or more {lat, lon, label}
    candidates.

    `limit` (1..10) caps the number of candidates — the address-search UI uses
    >1 for an autocomplete dropdown; a single-result caller passes 1.

    `viewbox` (optional) is a Nominatim bounding box "lon_min,lat_min,lon_max,
    lat_max". When given it biases results toward that box; with `bounded=1` it
    RESTRICTS results to the box (so a search over Washington State doesn't
    surface a match in France). Passed straight through to Nominatim.

    Returns:
      200 {"results": [{"lat": float, "lon": float, "label": str}, ...]}
      400 {"error": "empty_query"}      blank query
      502 {"error": "upstream_error"}   upstream failed
    A no-match is a 200 with an empty `results` list (not a 404) so the client
    can show "no matches" without treating it as an error.
    """
    query = (q or "").strip()
    if not query:
        return JSONResponse({"error": "empty_query"}, status_code=400)

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 1
    limit = max(1, min(limit, 10))

    # Validate the viewbox: exactly four comma-separated finite numbers. A
    # malformed value is ignored (unbounded search) rather than erroring.
    vb = (viewbox or "").strip()
    valid_viewbox = None
    if vb:
        parts = vb.split(",")
        if len(parts) == 4:
            try:
                nums = [float(p) for p in parts]
                valid_viewbox = ",".join(repr(n) for n in nums)
            except ValueError:
                valid_viewbox = None
    bounded_flag = 1 if (valid_viewbox and int(bounded or 0)) else 0

    cache_key = (limit, query.lower(), valid_viewbox or "", bounded_flag)
    cached = _cache_get(cache_key)
    if cached is not None:
        return JSONResponse({"results": cached})

    # Cache MISS -> we must hit Nominatim. Enforce the <=1 req/s upstream cap
    # (cache hits above bypass this, since they make no upstream call). If the
    # rate-limit backlog would make an interactive geocode wait too long, shed
    # the request with 503 rather than queueing unboundedly.
    if not _rate_gate.acquire(_MAX_WAIT):
        logger.info("geocode rate-limited (upstream slot > %.1fs out)", _MAX_WAIT)
        return JSONResponse({"error": "rate_limited"}, status_code=503)

    # `requests` ships with the lib package (installed in the same venv).
    import requests

    params = {"format": "jsonv2", "limit": limit, "q": query}
    if valid_viewbox:
        params["viewbox"] = valid_viewbox
        if bounded_flag:
            params["bounded"] = 1
    try:
        resp = requests.get(
            _NOMINATIM_URL,
            params=params,
            headers={"User-Agent": _user_agent(), "Accept": "application/json"},
            timeout=_TIMEOUT,
        )
    except requests.RequestException:
        logger.warning("geocode upstream request failed", exc_info=True)
        return JSONResponse({"error": "upstream_error"}, status_code=502)

    if not resp.ok:
        logger.warning("geocode upstream status %s", resp.status_code)
        return JSONResponse({"error": "upstream_error"}, status_code=502)

    try:
        data = resp.json()
    except ValueError:
        return JSONResponse({"error": "upstream_error"}, status_code=502)

    if not isinstance(data, list):
        return JSONResponse({"error": "upstream_error"}, status_code=502)

    results = []
    for hit in data:
        try:
            lat = float(hit["lat"])
            lon = float(hit["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append({"lat": lat, "lon": lon,
                        "label": hit.get("display_name") or query})

    # Cache the (possibly empty) result list so repeated lookups — including
    # misses — don't re-hit the upstream.
    _cache_put(cache_key, results)
    return JSONResponse({"results": results})
