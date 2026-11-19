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

# Forecast hours to sample, per element TYPE:
#  * instantaneous (TMP, WIND, ...): hourly through day 3, 3-hourly to ~7 days.
#  * accumulation  (ASNOW, APCP):    a MIXED-cadence disjoint tiling —
#    1-HOUR blocks for f1..f48 then 6-HOUR blocks for f54..f180. NBM publishes
#    accumulation as contiguous blocks ending at each forecast hour; the
#    1-hour blocks (0-1, 1-2, ...) are available only through ~f48, and only
#    6-hour blocks (..-54, 54-60, ...) exist beyond that. Taking the 1-hour
#    blocks early and 6-hour blocks late tiles [0,180] with NO overlap and NO
#    gap (verified against the .idx), so the cumulative storm total is exact —
#    AND the series now starts at f1 (init+1h) with hourly resolution over the
#    first two days, instead of the old 6-hourly-only schedule that began at f6
#    and discarded the first six hours.
def _instant_hours():
    hrs = list(range(1, 73))                 # hourly, days 1-3
    hrs += list(range(75, 181, 3))           # 3-hourly, days 4-7 (to f180)
    return hrs


# Accumulation tiling boundary: 1-hour blocks are published through this fhour;
# beyond it only 6-hour blocks exist. (NBM ASNOW/APCP; verified via the idx.)
ACCUM_HOURLY_MAX = 48
ACCUM_CADENCE = 6                            # hours per block in the 6-hourly tail


def _accum_hours():
    # Hourly 1..48 (1-hour blocks), then 6-hourly 54..180 (6-hour blocks).
    # Together these tile [0,180] disjointly with no gaps/overlaps.
    hourly = list(range(1, ACCUM_HOURLY_MAX + 1))
    six = list(range(ACCUM_HOURLY_MAX + ACCUM_CADENCE, 181, ACCUM_CADENCE))
    return hourly + six


def _accum_interval_for(fhour):
    """The accumulation block LENGTH (hours) to request for a given accum
    fhour: 1h while hourly blocks are published (fhour <= ACCUM_HOURLY_MAX),
    else 6h. Matches the disjoint tiling produced by _accum_hours()."""
    return 1 if fhour <= ACCUM_HOURLY_MAX else ACCUM_CADENCE


def _forecast_hours(accumulate):
    return _accum_hours() if accumulate else _instant_hours()


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


def _http_get(url, rng=None, timeout=60):
    """GET `url` (optionally a byte range), returning raw bytes or None."""
    import requests
    headers = {"User-Agent": _nbm_user_agent()}
    if rng is not None:
        headers["Range"] = f"bytes={rng}"
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
    except Exception:
        logger.warning("NBM GET failed %s", url, exc_info=True)
        return None
    if r.status_code in (200, 206):
        return r.content
    return None


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


def _fetch_idx_lines(day, cc, fhour):
    """Return the .idx lines for one hour (cached in-process briefly)."""
    url = _idx_url(day, cc, fhour)
    with _IDX_LOCK:
        hit = _IDX_CACHE.get(url)
    if hit is not None and time.time() - hit[0] < NBM_TTL_SECONDS:
        return hit[1]
    data = _http_get(url)
    if data is None:
        return None
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

    data = _http_get(url, rng=range_header)
    if not data or data[:4] != b"GRIB":
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
    """Fetch (idx+range) and parse all forecast-hour records for `attribute`
    from the latest cycle; build the KDTree once and the (epoch, values) frame
    list. Memoized per attribute keyed on (cycle, n_frames). Returns the dict
    or None."""
    spec = ELEMENTS.get(attribute)
    if spec is None:
        return None
    cyc = latest_cycle()
    if cyc is None:
        return None
    day, cc = cyc
    var, level = spec["var"], spec["level"]
    accum = spec.get("accum", False)
    hours = _forecast_hours(accum)

    cache_key = (day, cc, attribute)
    with _GRID_CACHE_LOCK:
        cached = _GRID_CACHE.get(attribute)
        if cached is not None and cached["key"] == cache_key:
            return cached

    tree = None
    frames = []

    # Resolve each record to a fetch spec (idx read only; no body download).
    # Records already fresh on disk need no fetch; the rest are fetched
    # CONCURRENTLY in the single writer process via one ensure_binary_many RPC
    # (MAX_CONCURRENCY-wide) instead of one blocking RPC per record.
    ready_paths = []     # fresh-on-disk records to parse
    fetch_specs = []     # {url,path,rng,key} needing a body fetch
    for fhour in hours:
        want = _accum_interval_for(fhour) if accum else None
        kind, payload = _resolve_record_spec(day, cc, fhour, var, level, want)
        if kind == "fresh":
            ready_paths.append(payload)
        elif kind == "fetch":
            fetch_specs.append(payload)
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
            for fhour in hours:
                want = _accum_interval_for(fhour) if accum else None
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

    result = {"key": cache_key, "tree": tree, "frames": deduped}
    with _GRID_CACHE_LOCK:
        _GRID_CACHE[attribute] = result
    logger.warning("NBM parsed %s: %d frames (cycle %s/%s)",
                   attribute, len(deduped), day, cc)
    return result


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
    hours = _forecast_hours(spec.get("accum", False))
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
    # For ACCUMULATION, anchor the cumulative zero at the first block's START
    # (the cycle init), not at the first SAMPLED epoch (the first block's END).
    # epochs[0] is the end of the first block; its length is _accum_interval_for
    # of the first accum fhour (1h under the mixed-cadence schedule), so the
    # window/zero-anchor start is epochs[0] - that length. This makes the series
    # begin at init (f000) and the first block ramp from 0, recovering the
    # previously-dropped opening hours. For OBSERVATION, the first epoch is the
    # value's own time, so start stays at epochs[0].
    if accumulate:
        # First block's length comes from the accumulation schedule's first
        # fhour (1h under the mixed-cadence tiling); init = first epoch - that.
        first_fhour = _accum_hours()[0]
        first_block_h = _accum_interval_for(first_fhour)
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
