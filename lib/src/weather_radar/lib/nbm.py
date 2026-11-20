"""NBM (National Blend of Models) GRIB2 weather source.

A THIRD drop-in weather source alongside NDFD (lib/ndfd.py) and the
api.weather.gov JSON (/gridpoints). Selected by WR_WEATHER_SOURCE=nbm. Like
NDFD it produces the SAME fitted artifacts (per-lattice-point CubicSpline over
time + the (start,end,center,f,units,expiry) tuple) so the endpoints, bundle
builder, and client consume it UNCHANGED (see models.nbm_model).

WHY NBM (vs NDFD):
  * NDFD's public GRIB does NOT publish extended-range (day 4-7) snowfall/qpf
    — those elements 404 for VP.004-007, capping snow at ~3 days. NBM carries
    ASNOW/APCP out to ~f264 (11 days).
  * NBM is NOAA's operational statistical blend (~30 models) on a national
    ~2.5km CONUS grid — the authoritative gridded forecast.

DISTRIBUTION — the opposite layout of NDFD:
  * NDFD: one file PER ELEMENT holding ALL hours -> one download per element.
  * NBM : one file PER FORECAST HOUR holding ALL elements. So "one element
    across time" means touching MANY hourly files. We avoid downloading the
    ~100MB master per hour by using the published `.idx` sidecar + an HTTP
    Range request to pull ONLY the one element's GRIB record (~0.5MB). Fetched
    once nationally and cached/shared across all areas/clients, exactly like
    the NDFD/NAQFC caches.

PIPELINE (mirrors lib/ndfd.py):
  1. Resolve the latest available cycle (12/06/00/18 UTC) on the AWS public
     bucket. For each forecast hour in FORECAST_HOURS, fetch that hour's `.idx`,
     locate the DETERMINISTIC record for the element (empty trailing field — no
     `prob`/`% level`), and HTTP-Range fetch just that record to disk.
  2. Parse each single-message record with eccodes; build a grid index ONCE
     (constant across hours) for bulk nearest sampling. The index projects
     lat/lon with pyproj and computes the cell by arithmetic (O(1), no per-cell
     arrays; see lib.gridindex), falling back to a cKDTree over the grid's
     lat/lon only if the projection can't be validated.
  3. For an area, nearest-sample every lattice point across all fetched hours,
     fit a per-point CubicSpline over (epoch -> value), convert native GRIB
     units to display units. Accumulation elements (ASNOW/APCP) are per-interval
     like NDFD -> the accumulate=True path integrates a cumulative monotone total.

FRESHNESS: NBM runs hourly; AWS objects carry no usable Expires for our stat
check, so (like NDFD) we use a TTL from file mtime.
"""

import os
import re
import time
import logging
import threading
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Public, keyless NBM GRIB2 on AWS Open Data. {day}=YYYYMMDD, {cc}=cycle.
NBM_BASE = os.environ.get(
    "NBM_BASE_URL",
    "https://noaa-nbm-grib2-pds.s3.amazonaws.com",
)
NBM_REGION = os.environ.get("NBM_REGION", "co")   # co=CONUS, ak, hi, pr, gu, oc
CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
NBM_DIR = os.path.join(CACHE_ROOT, "weather-radar-nbm")


def _nbm_user_agent():
    """UA for NBM fetches (AWS NOMADS). AWS doesn't require one, but we keep
    parity with the NWS fetches and let NBM_USER_AGENT override; otherwise fall
    back to the shared NOAA UA (WR_CONTACT_EMAIL etc.), so one contact config
    covers every upstream."""
    override = os.environ.get("NBM_USER_AGENT")
    if override and override.strip():
        return override.strip()
    from .connection import noaa_user_agent
    return noaa_user_agent(warn=False)

# NBM runs 4x/day. Try newest first, fall back.
CYCLES = ("18", "12", "06", "00")

# TTL freshness (seconds): a record file younger than this is fresh.
NBM_TTL_SECONDS = int(os.environ.get("NBM_TTL_SECONDS", "3600"))
# Negative-cache window for a (cycle, hour, element) that 404s / has no record.
NBM_MISS_TTL_SECONDS = int(os.environ.get("NBM_MISS_TTL_SECONDS", "3600"))

# Forecast hours to sample. We take EVERY forecast hour NBM publishes for the
# cycle (discovered from the bucket listing; see _published_hours), so each
# point's temporal spline gets as many knots as the product offers:
#  * instantaneous (TMP, WIND, ...): every published hour. For NBM v5 CONUS
#    that's hourly f1..f48, 3-hourly to f192, 6-hourly to f264 (108 hours).
#  * accumulation (ASNOW, APCP): the finest gap-free tiling of the published
#    blocks — 1-HOUR blocks through the leading hourly run (f1..f48), then
#    6-HOUR blocks (48-54, 54-60, ...) out to the last hour. Only 6-hour blocks
#    tile past f48 (the 3-hourly files carry either no block or an isolated
#    1-hour one that can't be summed), so this is the most knots a correct
#    cumulative total can have (verified against the .idx).
# Elements published at fewer hours (e.g. VIS stops at f78) simply miss the
# rest; a per-hour miss is negative-cached and skipped.

# Fallback schedule (verified NBM v5 CONUS) for when the listing is
# unavailable: hourly to f48, 3-hourly to f192, 6-hourly to f264.
_DEFAULT_HOURS = (list(range(1, 49)) + list(range(51, 193, 3))
                  + list(range(198, 265, 6)))

# Default accumulation tiling boundary (end of the hourly run) and the block
# length used past it. The live boundary is derived from the published hours.
ACCUM_HOURLY_MAX = 48
ACCUM_CADENCE = 6                            # hours per block past the hourly run


def _instant_hours(hours=None):
    """Instantaneous schedule: every published forecast hour."""
    return sorted(set(_DEFAULT_HOURS if hours is None else hours))


def _accum_schedule(hours=None):
    """[(fhour, block_hours)] tiling [0, H] contiguously with the finest
    accumulation blocks available: 1-hour blocks while the published hours run
    hourly from f1, then ACCUM_CADENCE-hour blocks while those end on a
    published hour. Stops at the first gap so blocks never overlap or skip."""
    hset = set(_DEFAULT_HOURS if hours is None else hours)
    out = []
    t = 0
    while (t + 1) in hset:
        out.append((t + 1, 1))
        t += 1
    while (t + ACCUM_CADENCE) in hset:
        out.append((t + ACCUM_CADENCE, ACCUM_CADENCE))
        t += ACCUM_CADENCE
    return out


def _accum_hours(hours=None):
    return [fh for fh, _ in _accum_schedule(hours)]


def _accum_interval_for(fhour, hours=None):
    """The accumulation block LENGTH (hours) to request for an accum fhour in
    the schedule derived from `hours` (default: the fallback schedule)."""
    for fh, length in _accum_schedule(hours):
        if fh == fhour:
            return length
    return ACCUM_CADENCE


def _forecast_hours(accumulate, hours=None):
    return _accum_hours(hours) if accumulate else _instant_hours(hours)


# Back-compat: default (instantaneous) schedule. Tests may override.
FORECAST_HOURS = _instant_hours()

# GRIB fill / no-data sentinel handling: NBM uses a bitmap; eccodes returns the
# GRIB missingValue (often 9999) for masked cells. Treat very large values as
# fill (same guard magnitude as NDFD).
_FILL = 9999.0


# --- Element mapping --------------------------------------------------------
# Maps the app's kebab-case attribute -> how to find/read it from NBM:
#   var    : GRIB shortName token in the .idx 4th field (e.g. "TMP").
#   level  : the .idx 5th field (level/layer) to disambiguate (e.g. TMP exists
#            at both "surface" and "2 m above ground").
#   units  : app DISPLAY unit string (what the legend shows), AFTER convert().
#   convert: native-GRIB-value -> display-value (identity if None).
#   nonneg : clamp predictions to >= 0.
#   accum  : True for per-interval accumulation elements (ASNOW/APCP), matched
#            by "acc fcst" records; the ensemble integrates a cumulative total.
# Native units VERIFIED/assumed from NBM GRIB docs: TMP/DPT/APTMP=K, RH/TCDC=%,
# WIND/GUST=m/s, WDIR=deg, VIS=m, APCP=kg/m^2(==mm), ASNOW=m.
_K_TO_F = lambda k: (float(k) - 273.15) * 9.0 / 5.0 + 32.0
_MS_TO_MPH = lambda v: float(v) * 2.2369362920544
_MM_TO_IN = lambda v: float(v) / 25.4
_M_TO_IN = lambda v: float(v) * 39.3700787
_M_TO_MI = lambda v: float(v) / 1609.344

ELEMENTS = {
    "temperature":                  {"var": "TMP",   "level": "2 m above ground", "units": "degF",    "convert": _K_TO_F,    "nonneg": False, "accum": False},
    "dewpoint":                     {"var": "DPT",   "level": "2 m above ground", "units": "degF",    "convert": _K_TO_F,    "nonneg": False, "accum": False},
    "apparent-temperature":         {"var": "APTMP", "level": "2 m above ground", "units": "degF",    "convert": _K_TO_F,    "nonneg": False, "accum": False},
    "relative-humidity":            {"var": "RH",    "level": "2 m above ground", "units": "percent", "convert": None,       "nonneg": True,  "accum": False},
    "sky-cover":                    {"var": "TCDC",  "level": "surface",          "units": "percent", "convert": None,       "nonneg": True,  "accum": False},
    "wind-speed":                   {"var": "WIND",  "level": "10 m above ground","units": "mph",     "convert": _MS_TO_MPH, "nonneg": True,  "accum": False},
    "wind-gust":                    {"var": "GUST",  "level": "10 m above ground","units": "mph",     "convert": _MS_TO_MPH, "nonneg": True,  "accum": False},
    "wind-direction":               {"var": "WDIR",  "level": "10 m above ground","units": "degrees", "convert": None,       "nonneg": True,  "accum": False},
    "visibility":                   {"var": "VIS",   "level": "surface",          "units": "mi",      "convert": _M_TO_MI,   "nonneg": True,  "accum": False},
    "quantitative-precipitation":   {"var": "APCP",  "level": "surface",          "units": "in",      "convert": _MM_TO_IN,  "nonneg": True,  "accum": True},
    "snowfall-amount":              {"var": "ASNOW", "level": "surface",          "units": "in",      "convert": _M_TO_IN,   "nonneg": True,  "accum": True},
}


def supports(attribute):
    """True if `attribute` (kebab-case) has an NBM element mapping."""
    return attribute in ELEMENTS


# --- Cycle resolution -------------------------------------------------------

_CYCLE_CACHE = {"resolved": None, "at": 0.0}
_CYCLE_LOCK = threading.Lock()
# Re-resolve the latest cycle at most this often (NBM publishes ~hourly cycles
# but each takes time to appear; probing every request is wasteful).
_CYCLE_RECHECK_SECONDS = int(os.environ.get("NBM_CYCLE_RECHECK_SECONDS", "900"))


def _cycle_dir_url(day, cc):
    return f"{NBM_BASE}/blend.{day}/{cc}/core"


def _file_url(day, cc, fhour):
    return f"{_cycle_dir_url(day, cc)}/blend.t{cc}z.core.f{fhour:03d}.{NBM_REGION}.grib2"


def _idx_url(day, cc, fhour):
    return _file_url(day, cc, fhour) + ".idx"


_HTTP_SESSION = None
_HTTP_SESSION_LOCK = threading.Lock()
_HTTP_ATTEMPTS = 3


def _session():
    """One shared keep-alive HTTP session for NBM .idx / record GETs.

    Reusing pooled connections means a load of ~100 small .idx files opens a
    handful of connections (and does a handful of DNS lookups) instead of one
    per file — a burst of fresh lookups is what trips a single home-LAN
    resolver ("Temporary failure in name resolution")."""
    global _HTTP_SESSION
    with _HTTP_SESSION_LOCK:
        if _HTTP_SESSION is None:
            import requests
            from requests.adapters import HTTPAdapter
            s = requests.Session()
            n = max(_IDX_CONCURRENCY, 4) + 2
            adapter = HTTPAdapter(pool_connections=n, pool_maxsize=n,
                                  max_retries=0)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            _HTTP_SESSION = s
        return _HTTP_SESSION


def _http_fetch(url, rng=None, timeout=60):
    """GET `url` (optionally a byte range) with retries for TRANSIENT failures.

    Returns (status, content):
      ("ok", bytes)       200 / 206
      ("missing", None)   403 / 404 / 410 — the object really isn't there
      ("error", None)     network/DNS error, timeout, 429/5xx, or any other
                          unexpected status, after retrying
    Callers negative-cache only "missing"; an "error" is retried next load."""
    headers = {"User-Agent": _nbm_user_agent()}
    if rng is not None:
        headers["Range"] = f"bytes={rng}"
    last = None
    for attempt in range(_HTTP_ATTEMPTS):
        try:
            r = _session().get(url, headers=headers, timeout=timeout)
        except Exception as e:
            last = type(e).__name__
        else:
            if r.status_code in (200, 206):
                return "ok", r.content
            if r.status_code in (403, 404, 410):
                return "missing", None
            last = f"HTTP {r.status_code}"
            if r.status_code < 500 and r.status_code != 429:
                break                         # other 4xx: retrying won't help
        if attempt < _HTTP_ATTEMPTS - 1:
            time.sleep(0.5 * (attempt + 1))
    logger.warning("NBM GET failed %s (%s)", url, last)
    return "error", None


def _http_get(url, rng=None, timeout=60):
    """GET `url` (optionally a byte range), returning raw bytes or None."""
    status, content = _http_fetch(url, rng=rng, timeout=timeout)
    return content if status == "ok" else None


def _cycle_available(day, cc):
    """True if this cycle's core dir has at least the f001 CONUS idx."""
    return _http_get(_idx_url(day, cc, 1)) is not None


def latest_cycle(force=False):
    """Resolve (day, cc) of the newest available NBM cycle, cached briefly.
    Probes the current and prior day across CYCLES newest-first."""
    now = time.time()
    with _CYCLE_LOCK:
        if (not force and _CYCLE_CACHE["resolved"] is not None
                and now - _CYCLE_CACHE["at"] < _CYCLE_RECHECK_SECONDS):
            return _CYCLE_CACHE["resolved"]

    today = datetime.now(timezone.utc)
    for ago in (0, 1):
        day = (today - timedelta(days=ago)).strftime("%Y%m%d")
        for cc in CYCLES:
            # Don't consider a future cycle on today's date.
            if ago == 0 and int(cc) > today.hour:
                continue
            if _cycle_available(day, cc):
                resolved = (day, cc)
                with _CYCLE_LOCK:
                    _CYCLE_CACHE["resolved"] = resolved
                    _CYCLE_CACHE["at"] = now
                logger.warning("NBM latest cycle: blend.%s/%s", day, cc)
                return resolved
    return None


# --- Published forecast hours (bucket listing) ------------------------------

_HOURS_CACHE = {}        # (day, cc) -> (fetched_at, [fhours])
_HOURS_LOCK = threading.Lock()


def _list_published_hours(day, cc):
    """Forecast hours with a published .idx for this cycle + region, from the
    S3 bucket listing (ListObjectsV2, paginated). None if the listing isn't
    available (e.g. a mirror without S3 listing) or is empty."""
    import html
    import requests
    prefix = f"blend.{day}/{cc}/core/blend.t{cc}z.core.f"
    pat = re.compile(rf"\.f(\d{{3}})\.{re.escape(NBM_REGION)}\.grib2\.idx$")
    hours = set()
    token = None
    try:
        for _ in range(50):                       # page cap (~1000 keys/page)
            params = {"list-type": "2", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            r = requests.get(NBM_BASE + "/", params=params, timeout=30,
                             headers={"User-Agent": _nbm_user_agent()})
            if not r.ok:
                return None
            for key in re.findall(r"<Key>([^<]+)</Key>", r.text):
                m = pat.search(key)
                if m:
                    hours.add(int(m.group(1)))
            m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>",
                          r.text)
            if not m:
                break
            token = html.unescape(m.group(1))
    except Exception as e:
        logger.warning("NBM bucket listing failed for %s/%s (%s)", day, cc,
                       type(e).__name__)
        return None
    return sorted(hours) or None


def _published_hours(day, cc):
    """Every forecast hour NBM has published for (day, cc), refreshed at most
    every NBM_CYCLE_RECHECK_SECONDS (a fresh cycle is still uploading its later
    hours, so the list grows for a while). Falls back to the verified default
    schedule when the listing is unavailable."""
    key = (day, cc)
    now = time.time()
    with _HOURS_LOCK:
        hit = _HOURS_CACHE.get(key)
    if hit is not None and now - hit[0] < _CYCLE_RECHECK_SECONDS:
        return hit[1]
    hours = _list_published_hours(day, cc)
    if not hours:
        if hit is not None:
            return hit[1]                         # keep the last good listing
        return list(_DEFAULT_HOURS)
    with _HOURS_LOCK:
        _HOURS_CACHE[key] = (now, hours)
    return hours


# --- Negative cache for missing (cycle, hour, element) records --------------

_MISSING = {}
_MISSING_LOCK = threading.Lock()


def _miss_key(day, cc, fhour, var):
    return (day, cc, fhour, var)


def _missing_recent(key):
    with _MISSING_LOCK:
        retry_at = _MISSING.get(key)
    return retry_at is not None and time.time() < retry_at


def _mark_missing(key):
    with _MISSING_LOCK:
        _MISSING[key] = time.time() + NBM_MISS_TTL_SECONDS


# --- Index parsing + record fetch -------------------------------------------

# A wgrib2-style .idx line:
#   num:startbyte:d=YYYYMMDDHH:VAR:LEVEL:DESC:EXTRA
_IDX_CACHE = {}           # idx_url -> (mtime_fetched, [lines])
_IDX_LOCK = threading.Lock()

# Returned by _fetch_idx_lines when the .idx couldn't be fetched because of a
# transient problem (network/DNS/5xx), as opposed to None = it doesn't exist.
_TRANSIENT = object()


def _fetch_idx_lines(day, cc, fhour):
    """Return the .idx lines for one hour (cached in-process briefly), None if
    the .idx doesn't exist, or _TRANSIENT if it couldn't be fetched right now."""
    url = _idx_url(day, cc, fhour)
    with _IDX_LOCK:
        hit = _IDX_CACHE.get(url)
    if hit is not None and time.time() - hit[0] < NBM_TTL_SECONDS:
        return hit[1]
    status, data = _http_fetch(url)
    if status == "missing":
        return None
    if status != "ok" or data is None:
        return _TRANSIENT
    lines = data.decode("utf-8", "replace").splitlines()
    with _IDX_LOCK:
        _IDX_CACHE[url] = (time.time(), lines)
    return lines


def _find_record_range(lines, var, level, want_interval=None, fhour=None):
    """Locate the DETERMINISTIC record (var, level) in idx `lines` and return
    (start_byte, end_byte_or_None). Deterministic = empty trailing field (no
    `prob`/`% level`/`ens std dev`).

    Selection:
      * Instantaneous (want_interval None): the single '... N hour fcst' record.
      * Accumulation (want_interval = C, fhour given): the CONTIGUOUS block
        '(fhour-C)-fhour hour acc', so consecutive fetched hours tile the window
        with no overlap/gap. If that exact block is absent, fall back to the
        LONGEST available acc interval ending at fhour (best contiguous cover),
        else the longest acc interval overall."""
    exact = None               # exact (fh-C)-fh block
    end_at_fh = None           # (ilen, start, end) longest ending at fhour
    longest = None             # (ilen, start, end) longest overall
    inst = None                # instantaneous record
    for i, ln in enumerate(lines):
        p = ln.split(":")
        if len(p) < 6:
            continue
        if p[3] != var or p[4] != level:
            continue
        extra = p[6] if len(p) > 6 else ""
        if extra != "":          # skip prob / percentile / std-dev records
            continue
        desc = p[5]
        start = int(p[1])
        nxt = lines[i + 1].split(":")[1] if i + 1 < len(lines) else None
        end = (int(nxt) - 1) if nxt else None
        m = re.match(r"(\d+)-(\d+) hour acc", desc)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            ilen = b - a
            if want_interval is not None and fhour is not None:
                if b == fhour and (fhour - a) == want_interval:
                    exact = (start, end)
                if b == fhour and (end_at_fh is None or ilen > end_at_fh[0]):
                    end_at_fh = (ilen, start, end)
            if longest is None or ilen > longest[0]:
                longest = (ilen, start, end)
        else:
            inst = (start, end)

    if want_interval is None:
        return inst if inst else None
    if exact is not None:
        return exact
    if end_at_fh is not None:
        return end_at_fh[1], end_at_fh[2]
    if longest is not None:
        return longest[1], longest[2]
    return None


def _record_path(day, cc, fhour, var):
    return os.path.join(
        NBM_DIR, f"nbm.{day}.{cc}.f{fhour:03d}.{NBM_REGION}.{var}.grib2")


def _is_fresh(path):
    try:
        return (time.time() - os.path.getmtime(path)) < NBM_TTL_SECONDS
    except OSError:
        return False


def _resolve_record_spec(day, cc, fhour, var, level, want_interval=None):
    """Resolve one (hour, element) record to a fetch spec WITHOUT downloading
    the body. Returns one of:
      ("fresh", path)              already on disk & fresh — no fetch needed
      ("miss",  path_or_None)      negative-cached / idx missing — skip
      ("retry", path_or_None)      idx unreachable right now (transient) —
                                   skip this pass, NOT negative-cached
      ("fetch", {url,path,rng})    needs fetching (the byte range to GET)
    Used by the batched loader so all record bodies can be fetched
    concurrently in the writer; the per-record idx read still happens here."""
    path = _record_path(day, cc, fhour, var)
    if _is_fresh(path):
        return ("fresh", path)
    key = _miss_key(day, cc, fhour, var)
    if _missing_recent(key):
        return ("miss", path if os.path.exists(path) else None)
    lines = _fetch_idx_lines(day, cc, fhour)
    if lines is _TRANSIENT:
        return ("retry", path if os.path.exists(path) else None)
    if lines is None:
        _mark_missing(key)
        return ("miss", path if os.path.exists(path) else None)
    rng = _find_record_range(lines, var, level, want_interval, fhour)
    if rng is None:
        _mark_missing(key)
        return ("miss", path if os.path.exists(path) else None)
    start, end = rng
    range_header = f"{start}-{end}" if end is not None else f"{start}-"
    return ("fetch", {"url": _file_url(day, cc, fhour), "path": path,
                      "rng": range_header, "key": key})


def _validate_grib(path, key):
    """Post-fetch: confirm `path` holds a GRIB record; if not, drop it and
    negative-cache `key`. Returns the path on success, else None."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except OSError:
        magic = b""
    if magic == b"GRIB":
        return path
    try:
        os.unlink(path)
    except OSError:
        pass
    _mark_missing(key)
    return None


def _ensure_record(day, cc, fhour, var, level, want_interval=None):
    """Ensure one (hour, element) GRIB record is on disk; return its path or
    None. Uses the .idx + HTTP Range to fetch ONLY that record. For
    accumulation elements pass want_interval (the block length ending at fhour)
    so a contiguous tiling is selected."""
    path = _record_path(day, cc, fhour, var)
    if _is_fresh(path):
        return path
    key = _miss_key(day, cc, fhour, var)
    if _missing_recent(key):
        return path if os.path.exists(path) else None

    lines = _fetch_idx_lines(day, cc, fhour)
    if lines is _TRANSIENT:
        return path if os.path.exists(path) else None   # retry next load
    if lines is None:
        _mark_missing(key)
        return path if os.path.exists(path) else None
    rng = _find_record_range(lines, var, level, want_interval, fhour)
    if rng is None:
        _mark_missing(key)
        return path if os.path.exists(path) else None
    start, end = rng
    range_header = f"{start}-{end}" if end is not None else f"{start}-"
    url = _file_url(day, cc, fhour)

    # Preferred: the single cache-writer process owns the write (atomic, no
    # cross-worker races), consistent with the AQI/NDFD/gridpoints paths. NBM
    # uses age-based TTL freshness (S3 Expires isn't used), so stamp_expires=
    # False keeps mtime at write time; force=True because our _is_fresh above
    # already gated the refetch. The record is fetched via the HTTP byte range.
    try:
        from . import cachewriter
        ua = _nbm_user_agent()
        ok = cachewriter.ensure_binary(
            url, path, force=True, rng=range_header,
            headers={"User-Agent": ua}, stamp_expires=False)
        if ok and os.path.exists(path):
            # Validate it's a GRIB record (a bad range could yield non-GRIB).
            try:
                with open(path, "rb") as f:
                    magic = f.read(4)
            except OSError:
                magic = b""
            if magic == b"GRIB":
                return path
            # Non-GRIB payload: drop it and negative-cache.
            try:
                os.unlink(path)
            except OSError:
                pass
            _mark_missing(key)
            return None
        # Writer reachable but produced no file -> treat as a miss.
        if not os.path.exists(path):
            _mark_missing(key)
            return None
        return path
    except Exception:
        pass  # writer unavailable -> inline fallback below

    status, data = _http_fetch(url, rng=range_header)
    if status == "error":
        return path if os.path.exists(path) else None   # transient: retry later
    if status != "ok" or not data or data[:4] != b"GRIB":
        _mark_missing(key)
        return path if os.path.exists(path) else None

    os.makedirs(NBM_DIR, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError:
        logger.warning("NBM record write failed %s", path, exc_info=True)
        return None
    return path


# --- Parsed-grid cache (per worker) -----------------------------------------

_GRID_CACHE = {}          # attribute -> {"key", "tree", "frames":[(epoch,vals)]}
_GRID_CACHE_LOCK = threading.Lock()


def _parse_record(path, want_index=True):
    """Parse one single-message NBM GRIB2 record into (index_or_None, epoch,
    values). The grid index (lat/lon -> flat cell index; see lib.gridindex) is
    only built when `want_index` — the grid is identical for every record, so
    callers need it from just the FIRST parsed record and pass False after
    that (building it per record was pure waste)."""
    import numpy as np
    import eccodes as ec
    from .gridindex import index_from_message, uniform_scan_values

    with open(path, "rb") as f:
        gid = ec.codes_grib_new_from_file(f)
        if gid is None:
            return None, None, None
        try:
            index = index_from_message(gid) if want_index else None
            # NBM uses alternating-row scanning: put the values into the
            # uniform layout the grid index expects (see uniform_scan_values).
            vals = uniform_scan_values(
                np.asarray(ec.codes_get_values(gid), dtype=float),
                lambda k: ec.codes_get(gid, k))
            vd = ec.codes_get(gid, "validityDate")
            vt = ec.codes_get(gid, "validityTime")
            dt = datetime(vd // 10000, (vd // 100) % 100, vd % 100,
                          vt // 100, vt % 100, tzinfo=timezone.utc)
            return index, int(dt.timestamp()), vals
        finally:
            ec.codes_release(gid)


def _load_grid(attribute):
    """Fetch (idx+range) and parse every published forecast-hour record for
    `attribute` from the latest cycle; build the grid index once and the
    (epoch, values) frame list. Memoized per attribute, keyed on the cycle AND
    the fetch plan (so a cycle first loaded mid-upload is rebuilt once more of
    its hours are published). Returns the dict or None."""
    import concurrent.futures

    spec = ELEMENTS.get(attribute)
    if spec is None:
        return None
    cyc = latest_cycle()
    if cyc is None:
        return None
    day, cc = cyc
    var, level = spec["var"], spec["level"]
    accum = spec.get("accum", False)
    published = _published_hours(day, cc)
    if accum:
        plan = _accum_schedule(published)            # [(fhour, block_hours)]
    else:
        plan = [(fh, None) for fh in _instant_hours(published)]
    if not plan:
        return None

    cache_key = (day, cc, attribute, len(plan), plan[-1][0])
    with _GRID_CACHE_LOCK:
        cached = _GRID_CACHE.get(attribute)
        if (cached is not None and cached["key"] == cache_key
                and (cached.get("retry_at") is None
                     or time.time() < cached["retry_at"])):
            return cached

    tree = None
    frames = []

    # Resolve each record to a fetch spec (idx read only; no body download),
    # several hours at a time — a cold load reads ~100 small .idx files.
    # Records already fresh on disk need no fetch; the rest are fetched
    # CONCURRENTLY in the single writer process via one ensure_binary_many RPC
    # instead of one blocking RPC per record.
    with concurrent.futures.ThreadPoolExecutor(max_workers=_IDX_CONCURRENCY) as ex:
        resolved = list(ex.map(
            lambda p: _resolve_record_spec(day, cc, p[0], var, level, p[1]), plan))
    ready_paths = []     # fresh-on-disk records to parse
    fetch_specs = []     # {url,path,rng,key} needing a body fetch
    n_skipped = 0        # hours skipped for a TRANSIENT reason (retry soon)
    for kind, payload in resolved:
        if kind == "fresh":
            ready_paths.append(payload)
        elif kind == "fetch":
            fetch_specs.append(payload)
        elif kind == "retry":
            n_skipped += 1
            if payload:                 # an older copy on disk beats a gap
                ready_paths.append(payload)
        # "miss" -> skip (negative-cached / absent)

    if fetch_specs:
        ua = _nbm_user_agent()
        did_batch = False
        try:
            from . import cachewriter
            # NBM: age-TTL freshness (stamp_expires=False), force=True since
            # _resolve_record_spec already gated on _is_fresh.
            batch = [{"url": s["url"], "path": s["path"], "force": True,
                      "rng": s["rng"], "headers": {"User-Agent": ua},
                      "stamp_expires": False} for s in fetch_specs]
            oks = cachewriter.ensure_binary_many(batch)
            did_batch = True
            for s, ok in zip(fetch_specs, oks):
                p = s["path"]
                if ok and os.path.exists(p) and _validate_grib(p, s["key"]):
                    ready_paths.append(p)
                elif not ok:
                    n_skipped += 1
                # NOTE: do NOT _mark_missing on a failed fetch here. A failure
                # at this stage is almost always TRANSIENT (network/DNS/timeout
                # on a burst of concurrent range GETs), not a genuine "record
                # doesn't exist" (that's caught earlier in _resolve_record_spec
                # via the idx). Negative-caching a transient failure would leave
                # a hole in the forecast for NBM_MISS_TTL_SECONDS; instead we
                # just skip it this pass so the next load retries it.
        except cachewriter.WriterUnavailable:
            did_batch = False
        except Exception:
            logger.warning("NBM batched fetch failed; falling back to serial",
                           exc_info=True)
            did_batch = False
        if not did_batch:
            # Writer unavailable: fall back to the per-record path (which has
            # its own inline-download fallback).
            for fhour, want in plan:
                p = _ensure_record(day, cc, fhour, var, level, want_interval=want)
                if p and os.path.exists(p):
                    ready_paths.append(p)

    # Parse every ready record into a frame (dedup paths first).
    for path in dict.fromkeys(ready_paths):
        if not path or not os.path.exists(path):
            continue
        try:
            # The grid index is identical for every record: build it from the
            # first one that parses, skip it for the rest.
            t, epoch, vals = _parse_record(path, want_index=(tree is None))
        except Exception:
            logger.warning("NBM parse failed for %s", path, exc_info=True)
            continue
        if epoch is None:
            continue
        if tree is None:
            tree = t
        frames.append((epoch, vals))
    if tree is None or not frames:
        return None

    seen = set()
    deduped = []
    for epoch, vals in sorted(frames, key=lambda x: x[0]):
        if epoch in seen:
            continue
        seen.add(epoch)
        deduped.append((epoch, vals))

    # Cycle init (f000) epoch: the cumulative-total anchor for accumulation.
    init = int(datetime.strptime(day, "%Y%m%d")
               .replace(hour=int(cc), tzinfo=timezone.utc).timestamp())
    result = {"key": cache_key, "tree": tree, "frames": deduped, "init": init,
              # A grid with hours skipped for transient reasons is only kept
              # briefly, so those hours are retried instead of staying missing
              # for the rest of the cycle.
              "retry_at": (time.time() + _INCOMPLETE_RETRY_SECONDS
                           if n_skipped else None)}
    with _GRID_CACHE_LOCK:
        _GRID_CACHE[attribute] = result
    if n_skipped:
        logger.warning("NBM parsed %s: %d frames (cycle %s/%s), %d hour(s) "
                       "unavailable right now; retrying in %ds", attribute,
                       len(deduped), day, cc, n_skipped, _INCOMPLETE_RETRY_SECONDS)
    else:
        logger.warning("NBM parsed %s: %d frames (cycle %s/%s)",
                       attribute, len(deduped), day, cc)
    return result


_IDX_CONCURRENCY = int(os.environ.get("NBM_IDX_CONCURRENCY", "8"))
_INCOMPLETE_RETRY_SECONDS = int(os.environ.get("NBM_INCOMPLETE_RETRY_SECONDS", "60"))


def cycle_token(attribute):
    """Monotonic DATA-VERSION for NBM bundles: the newest cycle's init epoch
    plus 0.01 per published forecast hour, or 0.0 if unknown. It increases when
    a newer cycle appears or when more hours of the current cycle land (so a
    bundle built mid-upload is superseded), and — unlike a file mtime — does NOT
    change when the same data is merely re-fetched or re-stamped."""
    spec = ELEMENTS.get(attribute)
    if spec is None:
        return 0.0
    cyc = latest_cycle()
    if cyc is None:
        return 0.0
    day, cc = cyc
    try:
        init = datetime.strptime(day + cc, "%Y%m%d%H").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0
    n = len(_forecast_hours(spec.get("accum", False), _published_hours(day, cc)))
    return init + 0.01 * n


def grid_mtime(attribute):
    """Newest on-disk mtime of the records backing `attribute`, or 0.0 — the
    freshness signal for NBM bundles (analog of ndfd.grid_mtime)."""
    spec = ELEMENTS.get(attribute)
    if spec is None:
        return 0.0
    cyc = latest_cycle()
    if cyc is None:
        return 0.0
    day, cc = cyc
    var = spec["var"]
    hours = _forecast_hours(spec.get("accum", False), _published_hours(day, cc))
    newest = 0.0
    for fhour in hours:
        p = _record_path(day, cc, fhour, var)
        try:
            m = os.path.getmtime(p)
            if m > newest:
                newest = m
        except OSError:
            continue
    return newest


# --- Area fitting (stage 1 time; stage 2 space via fit_area_dense) ----------

def _fit_column(raw, epochs, start_epoch, convert, accumulate):
    """Per-point temporal fit — delegates to the SHARED interp.fit_column so
    NBM/NDFD/AQI fitting semantics stay identical (see lib.interp)."""
    from . import interp
    return interp.fit_column(raw, epochs, start_epoch, convert, accumulate,
                             fill=_FILL)


def _area_block(index, area):
    """The block of native grid cells covering the area's box, using the grid
    index's box query. Returns:
      a GridBlock          -> use exactly these cells
      _NO_OVERLAP          -> the box is outside the grid (no data there)
      None                 -> box query unavailable (KD-tree index, or an area
                              without bounds); caller falls back to the
                              nearest-cell lattice."""
    box_cells = getattr(index, "box_cells", None)
    if box_cells is None:
        return None
    try:
        b = area.requested_bounds
    except AttributeError:
        return None
    try:
        block = box_cells(b["s"], b["n"], b["w"], b["e"])
    except Exception:
        logger.warning("NBM box query failed; falling back to the lattice",
                       exc_info=True)
        return None
    return _NO_OVERLAP if block is None else block


_NO_OVERLAP = object()


def fit_area(area, attribute, accumulate=False):
    """Resolve an area's NBM field into per-cell splines over the forecast
    series (epoch -> display value). Same return shape as ndfd.fit_area:
    {start, end, points:[MapCoordinate], splines:[...], units, nonneg} or None.
    Points with < 2 valid samples are dropped.

    With the pyproj grid index, the points are the ACTUAL NBM grid cells in the
    block covering the area's box (no nearest-neighbor resampling, so no
    duplicated cells), and the fit also carries a "block" entry so
    fit_area_dense can densify in grid space. With the KD-tree fallback, the
    points are the area's lat/lon lattice snapped to the nearest cell (the
    previous behavior)."""
    import numpy as np
    from .area import MapCoordinate

    spec = ELEMENTS.get(attribute)
    if spec is None:
        return None
    grid = _load_grid(attribute)
    if grid is None or grid["tree"] is None or not grid["frames"]:
        return None

    block = _area_block(grid["tree"], area)
    if block is _NO_OVERLAP:
        return None
    if block is not None:
        idx = block.flat
        pt_lats, pt_lons = block.lats, block.lons
    else:
        coords = list(area.map_coordinates)
        if not coords:
            return None
        pts = np.array([[c.lat, c.lon] for c in coords])
        _, idx = grid["tree"].query(pts, k=1)
        pt_lats, pt_lons = pts[:, 0], pts[:, 1]

    epochs = np.array([e for e, _ in grid["frames"]], dtype=float)
    # Gather only the selected cells from each frame. (Stacking whole frames
    # first would copy every ~3.7M-cell grid for every fit.)
    series = np.vstack([v[idx] for _, v in grid["frames"]])

    convert = spec["convert"]
    # For ACCUMULATION, anchor the cumulative zero at the cycle init (f000) —
    # the start of the first block — not at the first SAMPLED epoch (the first
    # block's end), so the series begins at init and the first block ramps
    # from 0. The grid records the init epoch; a grid without it (tests) falls
    # back to "first epoch minus the first block's length". For OBSERVATION,
    # the first epoch is the value's own time, so start stays at epochs[0].
    if accumulate:
        init = grid.get("init")
        if init is not None and init < epochs[0]:
            start_epoch = int(init)
        else:
            first_block_h = _accum_interval_for(_accum_hours()[0])
            start_epoch = int(epochs[0]) - first_block_h * 3600
    else:
        start_epoch = int(epochs[0])
    out_points = []
    out_splines = []
    kept = []          # block position (row-major) of each kept point
    for col in range(series.shape[1]):
        sp = _fit_column(series[:, col], epochs, start_epoch, convert, accumulate)
        if sp is None:
            continue
        out_points.append(MapCoordinate(lat=float(pt_lats[col]),
                                        lon=float(pt_lons[col])))
        out_splines.append(sp)
        kept.append(col)

    if not out_splines:
        return None
    fit = {
        "start": start_epoch,
        "end": int(epochs[-1]),
        "points": out_points,
        "splines": out_splines,
        "units": spec["units"],
        "nonneg": spec["nonneg"],
        # The times the data actually exists at (plus the zero anchor for an
        # accumulation), so a re-fit can use exactly these as knots.
        "knots": (np.concatenate(([float(start_epoch)], epochs))
                  if accumulate and start_epoch < epochs[0] else epochs),
    }
    if block is not None:
        fit["block"] = {"block": block, "positions": np.asarray(kept)}
    return fit


WR_NBM_UPSAMPLE = "WR_NBM_UPSAMPLE"
_DENSE_KNOT_CAP = 200


def fit_area_dense(area, attribute, accumulate=False, upsample=None):
    """Spatially densified fit (shared lib.interp two-stage), identical in
    spirit to ndfd.fit_area_dense. Pass-through to fit_area at upsample<=1."""
    import numpy as np
    from scipy.interpolate import CubicSpline, PchipInterpolator
    from .area import MapCoordinate
    from . import interp

    k = interp.upsample_env(WR_NBM_UPSAMPLE) if upsample is None else max(int(upsample), 1)
    fit = fit_area(area, attribute, accumulate=accumulate)
    if k <= 1 or fit is None:
        return fit

    pts = fit["points"]
    if not pts:
        return fit
    # Re-fit at the ACTUAL forecast times (every published hour) rather than an
    # evenly spaced grid: exact at every data point, and no cap that would thin
    # out the hourly opening days over the full ~11-day horizon.
    knots = fit.get("knots")
    if knots is not None and np.asarray(knots).size >= 2:
        epochs = np.unique(np.asarray(knots, dtype=float))
    else:
        span = max(1, int(fit["end"]) - int(fit["start"]))
        n_knots = min(_DENSE_KNOT_CAP, max(2, span // 3600 + 1))
        epochs = np.linspace(fit["start"], fit["end"], num=n_knots, dtype=float)

    def _refit(values, ep):
        v = np.asarray(values, dtype=float)
        good = np.isfinite(v)
        if good.sum() < 2:
            return None
        try:
            return PchipInterpolator(ep[good], v[good]) if accumulate \
                else CubicSpline(ep[good], v[good])
        except Exception:
            return None

    blk = fit.get("block")
    if blk is not None:
        # Native NBM cells: densify in grid (row, col) space, where the block
        # is a regular grid.
        block = blk["block"]
        dense = interp.densify_block(block.shape, blk["positions"], epochs,
                                     fit["splines"], _refit, k, block.latlon)
    else:
        lats = np.array([p.lat for p in pts], dtype=float)
        lons = np.array([p.lon for p in pts], dtype=float)
        dense = interp.densify_fit(lats, lons, epochs, fit["splines"], _refit, k)
    if dense is None:
        return fit
    d_lats, d_lons, d_splines = dense
    out_points, out_splines = [], []
    for la, lo, sp in zip(d_lats, d_lons, d_splines):
        if sp is None:
            continue
        out_points.append(MapCoordinate(lat=float(la), lon=float(lo)))
        out_splines.append(sp)
    if not out_splines:
        return fit
    return {
        "start": fit["start"], "end": fit["end"],
        "points": out_points, "splines": out_splines,
        "units": fit["units"], "nonneg": fit["nonneg"],
    }
