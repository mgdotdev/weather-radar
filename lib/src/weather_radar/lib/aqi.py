"""NOAA/EPA National Air Quality Forecast Capability (NAQFC) air-quality layer.

This is the AQI analog of the NOAA weather layers: a dense, gridded, HOURLY
forecast (ozone + PM2.5) out to ~72 hours, published twice daily by NOAA on
NOMADS as GRIB2. Unlike AirNow (sparse city points / daily categories), NAQFC
is a numerical model field — so it drops into the app's existing gridded,
animated, hover/pin/scrubber pipeline.

Pipeline
--------
1. Download the latest NAQFC cycle's hourly PM2.5 (or ozone) GRIB2 from NOMADS
   (keyless, public). One national file holds all ~72 forecast-hour messages.
2. For each message (= one forecast hour), read the value array + the grid's
   lat/lon arrays ONCE. The grid is Lambert Conformal; rather than project, we
   bulk-nearest-sample via a scipy cKDTree built once per file (the grid is
   constant across messages). `codes_grib_find_nearest` is ~500ms/point and
   unusable for bulk; the KDTree is ~1us/point.
3. Convert the raw concentration (PM2.5 ug/m3, ozone ppb) to the EPA AQI index
   via the standard piecewise breakpoints, so the client colors it on the AQI
   scale and hover/pins show a real AQI number.
4. Each message becomes a frame at its GRIB `validityDate/validityTime` (UTC),
   feeding the normal frame cache / scrubber.

Caching
-------
The GRIB file is large (~70MB) but national — one download serves every area
and client. It's cached on disk per (cycle, pollutant); parsed grids + KDTree
are memoized in-process per worker. Refetched when a newer cycle appears.
"""

import os
import re
import bz2
import math
import time
import logging
import threading
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

NOMADS_BASE = os.environ.get(
    "NAQFC_BASE_URL",
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aqm/prod",
)
# NAQFC historical archive (NOAA Open Data on AWS). Unlike NOMADS (~2-day
# retention), this keeps every operational cycle back to 2020-01-01, keyed by
# model version / sector / date / cycle. We use it ONLY to backfill a cycle's
# trailing hours for NowCast (NOMADS alone can't reach far enough back when the
# requested "now" is near a cycle's issuance). AQMv7 / CS(=CONUS) is the live
# line since 2024-05-14 — the same products + 227 grid NOMADS serves.
NAQFC_ARCHIVE_BASE = os.environ.get(
    "NAQFC_ARCHIVE_BASE_URL",
    "https://noaa-nws-naqfc-pds.s3.amazonaws.com",
)
NAQFC_ARCHIVE_VERSION = os.environ.get("NAQFC_ARCHIVE_VERSION", "AQMv7")
NAQFC_ARCHIVE_SECTOR = os.environ.get("NAQFC_ARCHIVE_SECTOR", "CS")  # CONUS
CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
NAQFC_DIR = os.path.join(CACHE_ROOT, "weather-radar-naqfc")

# NAQFC runs at 06 and 12 UTC. We try the newest cycle first and fall back.
CYCLES = ("12", "06")

# NowCast trailing window: the EPA PM2.5 NowCast weights the most recent 12
# hourly concentrations. We need at least this many trailing hours (ending at
# the evaluated hour) to compute it; fewer -> fall back to the raw hourly AQI.
NOWCAST_HOURS = 12

# When the on-disk GRIB's Expires has passed but the SAME cycle is still the
# newest available (NAQFC only publishes twice daily, but Expires is ~4h), we
# extend the local freshness stamp by this long instead of re-downloading the
# identical file. Bounds how often we re-probe NOMADS for a newer cycle in the
# expired-but-unchanged window. Default 30 min.
NAQFC_RECHECK_SECONDS = int(os.environ.get("NAQFC_RECHECK_SECONDS", "1800"))

# Pollutant -> (NOMADS file infix, concentration->AQI converter key).
# `_bc` = bias-corrected (model nudged toward observations); preferred.
POLLUTANTS = {
    "pm25": "ave_1hr_pm25_bc",
    "ozone": "ave_1hr_o3_bc",
}

# NowCast variants are the SAME underlying field (same NOMADS infix / 227
# grid) with a different time-aggregation applied at fit time. They're surfaced
# as distinct public pollutant ids ("pm25-nowcast", "ozone-nowcast") so they
# flow through the existing `pollutant` query param, bundle keys, and catalog
# without new endpoints. `_parse_pollutant` maps them back to a base + flag.
_NOWCAST_SUFFIX = "-nowcast"


def _parse_pollutant(pollutant):
    """Split a public pollutant id into (base, nowcast: bool).

    "pm25"          -> ("pm25", False)
    "pm25-nowcast"  -> ("pm25", True)
    Unknown ids pass through as (pollutant, False) so callers' existing
    membership checks (`pollutant in POLLUTANTS`) behave unchanged for them."""
    p = (pollutant or "").strip().lower()
    if p.endswith(_NOWCAST_SUFFIX):
        return p[: -len(_NOWCAST_SUFFIX)], True
    return p, False

GRID_ID = "227"  # CONUS ~5km Lambert grid

# --- EPA AQI breakpoints ----------------------------------------------------
# AQI = (Ih-Il)/(Ch-Cl) * (C-Cl) + Il, over the band containing truncated C.
# PM2.5 uses the 2024-updated breakpoints (ug/m3, truncate to 0.1).
# Ozone uses 8-hour ppb bands here as a reasonable default for the 1hr field
# (NAQFC also ships an 8hr product; see note in value conversion).
_PM25_BP = [
    (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500),
]
_OZONE_BP_PPB = [  # 8-hour ozone, ppb
    (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
    (86, 105, 151, 200), (106, 200, 201, 300),
]


def _aqi_from_pm25(c):
    if c is None or c < 0 or c >= 9999:
        return None
    c = math.floor(c * 10) / 10.0
    for Cl, Ch, Il, Ih in _PM25_BP:
        if Cl <= c <= Ch:
            return round((Ih - Il) / (Ch - Cl) * (c - Cl) + Il)
    return 500  # above the top band


def _aqi_from_ozone(c):
    if c is None or c < 0 or c >= 9999:
        return None
    c = math.floor(c)  # ppb, integer truncation
    for Cl, Ch, Il, Ih in _OZONE_BP_PPB:
        if Cl <= c <= Ch:
            return round((Ih - Il) / (Ch - Cl) * (c - Cl) + Il)
    return 300  # ozone AQI not defined by 8hr above ~300; clamp


_CONVERTER = {"pm25": _aqi_from_pm25, "ozone": _aqi_from_ozone}


def _bands_to_aqi(c, bands, truncate, top):
    """Vectorized twin of _aqi_from_pm25/_aqi_from_ozone: concentrations `c`
    (any shape) -> AQI as float64, NaN where the scalar returns None (negative,
    >= 9999, or non-finite). Same inclusive bands, truncation, top-band value
    and round-half-even as the scalar versions."""
    import numpy as np
    c = np.asarray(c, dtype=float)
    ok = np.isfinite(c) & (c >= 0) & (c < 9999)
    t = np.where(ok, truncate(np.where(ok, c, 0.0)), 0.0)
    res = np.full(c.shape, float(top))          # above the top band
    done = np.zeros(c.shape, dtype=bool)
    for Cl, Ch, Il, Ih in bands:
        m = ok & ~done & (t >= Cl) & (t <= Ch)
        res = np.where(m, np.round((Ih - Il) / (Ch - Cl) * (t - Cl) + Il), res)
        done |= m
    return np.where(ok, res, np.nan)


def _aqi_from_pm25_array(c):
    import numpy as np
    return _bands_to_aqi(c, _PM25_BP, lambda x: np.floor(x * 10) / 10.0, 500)


def _aqi_from_ozone_array(c):
    import numpy as np
    return _bands_to_aqi(c, _OZONE_BP_PPB, np.floor, 300)


_CONVERTER_ARRAY = {"pm25": _aqi_from_pm25_array, "ozone": _aqi_from_ozone_array}


# --- NowCast (EPA weighted multi-hour concentration) ------------------------
# Consumer AQI (AirNow, Apple Weather) reports NowCast, not a single hour: a
# weighted average over the trailing 12 hourly CONCENTRATIONS that damps
# short-term spikes. Computed on concentration, THEN converted to AQI.
#
# Algorithm (EPA; same shape for PM2.5 and ozone, differing only in the band
# table the resulting concentration is run through):
#   over the last N hours c[0]=most-recent .. c[N-1]=oldest:
#     range = max(c) - min(c); w* = 1 - range/max(c)
#     weight = clamp(w*, 0.5, 1)
#     nowcast = Σ weight**i · c[i]  /  Σ weight**i   (i skips missing hours)
# At least 2 of the most recent 3 hours must be present, else NowCast is
# undefined for that hour (we return None -> caller omits / falls back).

def _nowcast_concentration(recent):
    """`recent` = concentrations ordered MOST-RECENT first (len up to
    NOWCAST_HOURS). None = missing hour. Returns the NowCast concentration, or
    None if it can't be computed per the EPA data-sufficiency rule."""
    vals = list(recent)[:NOWCAST_HOURS]
    # Data sufficiency: need >=2 of the 3 most recent hours.
    recent3 = [v for v in vals[:3] if v is not None and 0 <= v < 9999]
    if len(recent3) < 2:
        return None
    present = [v for v in vals if v is not None and 0 <= v < 9999]
    if len(present) < 2:
        return None
    cmax = max(present)
    cmin = min(present)
    if cmax <= 0:
        return 0.0
    w = 1.0 - (cmax - cmin) / cmax
    if w < 0.5:
        w = 0.5
    num = 0.0
    den = 0.0
    for i, v in enumerate(vals):
        if v is None or v < 0 or v >= 9999:
            continue
        wi = w ** i
        num += wi * v
        den += wi
    if den <= 0:
        return None
    return num / den


def _nowcast_matrix(conc, chunk=2048):
    """Vectorized twin of _nowcast_concentration over a whole (hours, points)
    concentration matrix: for every hour the NowCast of the trailing
    NOWCAST_HOURS hours (hours before the series start don't exist), NaN where
    the scalar returns None. Same EPA rules and the same k-ascending summation
    order as the scalar loop. Points are processed in chunks to bound memory
    (the window cube is hours x NOWCAST_HOURS x chunk)."""
    import numpy as np
    conc = np.asarray(conc, dtype=float)
    n_h, n_p = conc.shape
    out = np.full((n_h, n_p), np.nan)
    ks = range(NOWCAST_HOURS)
    for lo in range(0, n_p, chunk):
        c = conc[:, lo:lo + chunk]
        good = np.isfinite(c) & (c >= 0) & (c < 9999)
        v = np.where(good, c, np.nan)
        # win[k][i] = value k hours before hour i (NaN before the series start).
        win = []
        for k in ks:
            w_k = np.full(v.shape, np.nan)
            if k < n_h:
                w_k[k:] = v[:n_h - k]
            win.append(w_k)
        pres = [~np.isnan(w_k) for w_k in win]
        n3 = sum(pres[k].astype(int) for k in range(min(3, NOWCAST_HOURS)))
        n_all = sum(p.astype(int) for p in pres)
        cmax = np.full(v.shape, -np.inf)
        cmin = np.full(v.shape, np.inf)
        for k in ks:
            cmax = np.where(pres[k], np.maximum(cmax, win[k]), cmax)
            cmin = np.where(pres[k], np.minimum(cmin, win[k]), cmin)
        with np.errstate(divide="ignore", invalid="ignore"):
            wgt = np.maximum(1.0 - (cmax - cmin) / cmax, 0.5)
            num = np.zeros(v.shape)
            den = np.zeros(v.shape)
            for k in ks:
                wk = wgt ** k
                num += np.where(pres[k], wk * np.where(pres[k], win[k], 0.0), 0.0)
                den += np.where(pres[k], wk, 0.0)
            nc = np.where(cmax <= 0, 0.0, num / den)
        enough = (n3 >= 2) & (n_all >= 2)
        out[:, lo:lo + chunk] = np.where(enough, nc, np.nan)
    return out


# --- Cycle discovery + download ---------------------------------------------

def _latest_cycle():
    """Return (yyyymmdd, cycle) for the newest NAQFC run that exists on NOMADS,
    trying today's 12z, today's 06z, then yesterday's. Returns None if none
    reachable."""
    import requests
    from .connection import noaa_user_agent
    _ua = {"User-Agent": noaa_user_agent(warn=False)}  # NOMADS, not api.weather.gov
    now = datetime.now(timezone.utc)
    for day_offset in (0, 1):
        day = (now - timedelta(days=day_offset)).strftime("%Y%m%d")
        for cyc in CYCLES:
            url = f"{NOMADS_BASE}/aqm.{day}/{cyc}/"
            try:
                r = requests.get(url, timeout=15, headers=_ua)
                if r.ok and f"aqm.t{cyc}z" in r.text:
                    return day, cyc
            except Exception:
                continue
    return None


def _grib_path(day, cyc, pollutant):
    return os.path.join(NAQFC_DIR, f"aqm.{day}.t{cyc}z.{pollutant}.{GRID_ID}.grib2")


def _grib_url(day, cyc, pollutant):
    infix = POLLUTANTS[pollutant]
    return f"{NOMADS_BASE}/aqm.{day}/{cyc}/aqm.t{cyc}z.{infix}.{GRID_ID}.grib2"


def _prev_cycle(day, cyc):
    """The NAQFC cycle immediately BEFORE (day, cyc). Cycles are 06z then 12z
    each day, so the predecessor of 12z is the same day's 06z, and the
    predecessor of 06z is the PRIOR day's 12z. Returns (yyyymmdd, cycle)."""
    if cyc == "12":
        return day, "06"
    # cyc == "06" -> previous day's 12z
    d = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
    return (d - timedelta(days=1)).strftime("%Y%m%d"), "12"


def _archive_url(day, cyc, pollutant):
    """Historical-archive URL for a cycle's pollutant GRIB2. The archive repeats
    the date in the filename and nests under version/sector/date/cycle; the
    product infix + 227 grid are otherwise identical to NOMADS."""
    infix = POLLUTANTS[pollutant]
    return (f"{NAQFC_ARCHIVE_BASE}/{NAQFC_ARCHIVE_VERSION}/{NAQFC_ARCHIVE_SECTOR}"
            f"/{day}/{cyc}/aqm.t{cyc}z.{infix}.{day}.{GRID_ID}.grib2")


def _archive_path(day, cyc, pollutant):
    # Distinct local name (…​.archive.…) so an archived cycle never collides
    # with a same-day NOMADS download of the current cycle.
    return os.path.join(
        NAQFC_DIR, f"aqm.{day}.t{cyc}z.{pollutant}.archive.{GRID_ID}.grib2")


def _download_archive_grib(day, cyc, pollutant):
    """Fetch a PRIOR cycle's pollutant GRIB2 from the historical archive to
    disk (writer-owned), returning its path or None. Archived cycles are
    immutable, so once on disk it's permanently fresh — no Expires, no
    re-download. Used only to backfill NowCast trailing hours."""
    if pollutant not in POLLUTANTS:
        return None
    path = _archive_path(day, cyc, pollutant)
    if os.path.exists(path):
        return path
    url = _archive_url(day, cyc, pollutant)
    try:
        from . import cachewriter
        cachewriter.ensure_binary(url, path, False)
        return path if os.path.exists(path) else None
    except Exception:
        logger.warning("NAQFC archive fetch failed for %s", url, exc_info=True)
        return path if os.path.exists(path) else None


def _is_fresh(path):
    """True if `path` exists and its stamped Expires (mtime) is still in the
    future — a single stat, no network. The writer stamps mtime=Expires on
    download (cachewriter.ensure_binary), mirroring the NOAA JSON cache."""
    try:
        return os.path.getmtime(path) >= time.time()
    except OSError:
        return False


def _download_grib(day, cyc, pollutant, force=False):
    """Ensure one NAQFC pollutant GRIB2 for a cycle is on disk and fresh,
    returning its path (or None on failure).

    The actual download is delegated to the single CACHE-WRITER process
    (cachewriter.ensure_binary), so ALL disk writes stay in that one process —
    no cross-worker download races — and the file's mtime is stamped to the
    HTTP Expires there (cheap freshness). Skips work when already fresh (unless
    force). Falls back to an inline download only if the writer is unreachable.
    """
    path = _grib_path(day, cyc, pollutant)
    if not force and _is_fresh(path):
        return path
    url = _grib_url(day, cyc, pollutant)

    # Preferred: the writer owns the write.
    try:
        from . import cachewriter
        ok = cachewriter.ensure_binary(url, path, force)
        return path if (ok or os.path.exists(path)) else None
    except Exception:
        # Writer unavailable (e.g. tests, or before it's spawned): inline
        # fallback so AQI still works, download direct to disk + stamp Expires.
        pass

    import requests
    from .connection import noaa_user_agent
    tmp = path + ".tmp"
    try:
        os.makedirs(NAQFC_DIR, exist_ok=True)
        with requests.get(url, timeout=120, stream=True,
                          headers={"User-Agent": noaa_user_agent(warn=False)}) as r:
            if not r.ok:
                logger.warning("NAQFC download %s -> HTTP %s", url, r.status_code)
                return path if os.path.exists(path) else None
            expires = r.headers.get("Expires")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(tmp, path)
        if expires:
            try:
                exp = datetime.strptime(expires, "%a, %d %b %Y %H:%M:%S %Z")
                ts = exp.replace(tzinfo=timezone.utc).timestamp()
                os.utime(path, (ts, ts))
            except Exception:
                pass
        logger.warning("NAQFC downloaded (inline) %s (%d bytes, expires %s)",
                       os.path.basename(path), os.path.getsize(path), expires)
        return path
    except Exception:
        logger.warning("NAQFC download failed for %s", url, exc_info=True)
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return None


# --- Parsed-grid cache (per worker) -----------------------------------------
# Memoizes, per GRIB file + its mtime: the KDTree over grid points + per-hour
# value arrays and validity epochs. Keyed by (path, mtime) so a REFRESHED file
# (new cycle, new mtime) re-parses and the stale entry is evicted.

_GRID_CACHE = {}          # path -> {"size", "tree", "frames":[(epoch, values)]}
_GRID_CACHE_LOCK = threading.Lock()

# Throttle NOMADS cycle probing: remember the resolved current path per
# pollutant and only re-probe when the on-disk file is no longer fresh (its
# stamped Expires has passed). Avoids 3-6 HTTP probes on every AQI request.
_CURRENT_PATH = {}        # pollutant -> path
_CURRENT_CYCLE = {}       # pollutant -> (yyyymmdd, cycle) of the current grid
_CURRENT_PATH_LOCK = threading.Lock()


def _load_grid(path, pollutant):
    """Parse a NAQFC GRIB2 into (KDTree, [(epoch, values)]) and memoize it by
    (path, SIZE). Keyed on the file SIZE, not its mtime: ensure_grid bumps the
    mtime forward to extend freshness in the expired-but-same-cycle window (so
    it doesn't re-download the identical file), and keying the parse cache on
    mtime would make every such bump force a needless re-parse of unchanged
    bytes. Size changes iff the content does (a new cycle writes a different
    file — and to a different path anyway), so size is the stable content
    signal here."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    with _GRID_CACHE_LOCK:
        cached = _GRID_CACHE.get(path)
        if cached is not None and cached["size"] == size:
            return cached

    import numpy as np
    import eccodes as ec
    from scipy.spatial import cKDTree

    frames = []
    tree = None
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                if tree is None:
                    lats = np.asarray(ec.codes_get_array(gid, "latitudes"))
                    lons = np.asarray(ec.codes_get_array(gid, "longitudes"))
                    lons = np.where(lons > 180, lons - 360, lons)
                    tree = cKDTree(np.column_stack([lats, lons]))
                vals = np.asarray(ec.codes_get_values(gid))
                vd = ec.codes_get(gid, "validityDate")  # yyyymmdd int
                vt = ec.codes_get(gid, "validityTime")  # HHMM int
                dt = datetime(vd // 10000, (vd // 100) % 100, vd % 100,
                              vt // 100, vt % 100, tzinfo=timezone.utc)
                frames.append((int(dt.timestamp()), vals))
            finally:
                ec.codes_release(gid)

    result = {"size": size, "tree": tree, "frames": frames}
    with _GRID_CACHE_LOCK:
        # Replace this path's entry (a different-size file supersedes it). Other
        # pollutants' paths are left alone.
        _GRID_CACHE[path] = result
    logger.warning("NAQFC parsed %s: %d hourly frames", os.path.basename(path),
                   len(frames))
    return result


def ensure_grid(pollutant="pm25"):
    """Ensure the current NAQFC grid for `pollutant` is on disk (fresh) and
    parsed; return the parsed-grid dict, or None if unavailable.

    Freshness is the HTTP Expires stamped onto the file's mtime (~4h, set by
    NOMADS). The fast path is: a remembered current file that's still fresh ->
    no network, no cycle probe, just reuse/parse-from-disk. Only when the file
    has expired do we re-probe NOMADS for the latest cycle and (re)download —
    which naturally picks up a newer cycle when one has published."""
    if pollutant not in POLLUTANTS:
        return None

    # Fast path: a known current file that's still fresh.
    with _CURRENT_PATH_LOCK:
        cur = _CURRENT_PATH.get(pollutant)
    if cur is not None and _is_fresh(cur):
        return _load_grid(cur, pollutant)

    # Stale/unknown: re-probe the latest cycle. NOTE: NAQFC publishes only
    # twice daily (06z/12z) but the HTTP Expires is only ~4h out, so there's a
    # multi-hour window where Expires has passed yet the SAME cycle is still the
    # newest. In that window we must NOT re-download the (identical) 68MB file
    # on every request — that was the bug. The cycle's path encodes day+cyc, so
    # if the resolved path already exists on disk, it's the same cycle we have:
    # just extend its freshness (bump mtime forward by the recheck interval) and
    # reuse it. A genuinely NEWER cycle resolves to a DIFFERENT path (not on
    # disk) and downloads normally.
    cyc = _latest_cycle()
    if cyc is None:
        # NOMADS unreachable: fall back to the last file we have, even if its
        # Expires passed (stale data beats no data).
        if cur is not None and os.path.exists(cur):
            return _load_grid(cur, pollutant)
        return None
    day, c = cyc
    resolved = _grib_path(day, c, pollutant)
    if os.path.exists(resolved):
        # Same cycle we already downloaded: no re-download. Push the freshness
        # stamp out by NAQFC_RECHECK_SECONDS so the next request's fast path
        # hits (no re-probe, no re-download) until it's worth re-checking for a
        # newer cycle. Does NOT change content, only the freshness mtime.
        try:
            new_mtime = time.time() + NAQFC_RECHECK_SECONDS
            os.utime(resolved, (new_mtime, new_mtime))
        except OSError:
            pass
        with _CURRENT_PATH_LOCK:
            _CURRENT_PATH[pollutant] = resolved
            _CURRENT_CYCLE[pollutant] = (day, c)
        return _load_grid(resolved, pollutant)

    path = _download_grib(day, c, pollutant)
    if path is None:
        return None
    with _CURRENT_PATH_LOCK:
        _CURRENT_PATH[pollutant] = path
        _CURRENT_CYCLE[pollutant] = (day, c)
    return _load_grid(path, pollutant)


# --- Area sampling ----------------------------------------------------------

def _cycle_epoch(day, cyc):
    try:
        return datetime.strptime(day + cyc, "%Y%m%d%H").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def cycle_token_in_use(pollutant):
    """Data-version token of the NAQFC cycle THIS process last loaded for the
    pollutant (the cycle's init epoch), or 0.0 if none yet. Call it right after
    a fit to label the frames it produced. Unlike the grid file's mtime — which
    ensure_grid bumps forward without the data changing — it only changes when
    a genuinely new cycle is loaded."""
    base, _ = _parse_pollutant(pollutant)
    with _CURRENT_PATH_LOCK:
        cyc = _CURRENT_CYCLE.get(base)
    return _cycle_epoch(*cyc) if cyc else 0.0


def disk_cycle_token(pollutant):
    """Token of the newest NAQFC cycle for the pollutant whose GRIB is on disk
    (a directory listing — no parse, no network), or 0.0. Same scale as
    cycle_token_in_use, so a reader in a worker that has loaded nothing can
    still tell whether a stored bundle matches the data the writer has."""
    base, _ = _parse_pollutant(pollutant)
    pat = re.compile(r"^aqm\.(\d{8})\.t(\d{2})z\.%s\.%s\.grib2$"
                     % (re.escape(base), re.escape(GRID_ID)))
    best = 0.0
    try:
        names = os.listdir(NAQFC_DIR)
    except OSError:
        return 0.0
    for n in names:
        m = pat.match(n)
        if m:
            best = max(best, _cycle_epoch(m.group(1), m.group(2)))
    return best


# --- Area sampling ----------------------------------------------------------

def _prior_frames(pollutant):
    """Parsed frames for the cycle immediately BEFORE the current one, from the
    historical archive — used to backfill NowCast's trailing hours when the
    evaluated hour sits near the current cycle's start. Returns a parsed-grid
    dict (same shape as _load_grid) or None. Best-effort: any failure (no
    current cycle known yet, archive unreachable) yields None and NowCast
    simply falls back to whatever current-cycle history it has."""
    with _CURRENT_PATH_LOCK:
        cur_cycle = _CURRENT_CYCLE.get(pollutant)
    if not cur_cycle:
        return None
    pday, pcyc = _prev_cycle(*cur_cycle)
    path = _download_archive_grib(pday, pcyc, pollutant)
    if path is None:
        return None
    return _load_grid(path, pollutant)


def _combined_series(area, pollutant):
    """Build the hourly CONCENTRATION series for an area's lattice points,
    concatenating the prior archived cycle's frames (for NowCast trailing
    hours) ahead of the current cycle's, de-duplicated by epoch and sorted
    ascending.

    Returns (epochs, series, points, idx, grid) where:
      epochs : (n_hours,) float epoch seconds, ascending, unique
      series : (n_hours, n_points) concentrations at the lattice points
      points : [[lon,lat], ...] wire order
      idx    : nearest-cell index per lattice point (current grid's KDTree)
      grid   : the current parsed grid (for start/end window)
    or None if the current grid is unavailable."""
    import numpy as np

    grid = ensure_grid(pollutant)
    if grid is None or grid["tree"] is None or not grid["frames"]:
        return None

    coords = list(area.map_coordinates)
    if not coords:
        return None
    pts = np.array([[c.lat, c.lon] for c in coords])
    _, idx = grid["tree"].query(pts, k=1)

    # Merge prior-cycle frames (if any) with the current cycle's, keyed by
    # epoch. Current-cycle values win on overlap (they're the newer forecast
    # for the same valid hour). Both share the identical 227 grid / cell order,
    # so the SAME `idx` samples both — no re-query against the prior tree.
    by_epoch = {}
    prior = _prior_frames(pollutant)
    if prior is not None and prior["frames"]:
        for e, v in prior["frames"]:
            by_epoch[int(e)] = v
    for e, v in grid["frames"]:
        by_epoch[int(e)] = v

    epochs_sorted = sorted(by_epoch)
    epochs = np.array(epochs_sorted, dtype=float)
    # Gather ONLY this area's cells from each frame. (Stacking the whole frames
    # first copied ~1.5M cells x ~144 hours — about 1.7 GB and ~1.5 s — to read
    # a few hundred points.)
    series = np.vstack([by_epoch[e][idx] for e in epochs_sorted])   # (n_hours, pts)
    points = [[c.lon, c.lat] for c in coords]
    return epochs, series, points, idx, grid


# --- Fit memo ---------------------------------------------------------------
# A fit is a pure function of (pollutant, the loaded cycle, the area's points).
# One page load asks for it several times — /aqi/range, every /aqi/stream fill
# request, the bundle builder — and each used to redo the whole per-point
# conversion + CubicSpline fit (and, before cells were gathered per frame, a
# ~1.7 GB frame stack). Memoizing it makes those one fit per (area, cycle).
# Entries are small (splines only) and hold NO reference to the parsed grid, so
# a superseded cycle's ~1 GB of frames can be freed as soon as _GRID_CACHE
# drops it.

_FIT_CACHE = {}           # key -> (pin, result), insertion-ordered
_FIT_CACHE_LOCK = threading.Lock()
_FIT_KEY_LOCKS = {}       # key -> Lock, so concurrent identical requests compute once
_FIT_CACHE_MAX = int(os.environ.get("WR_AQI_FIT_CACHE", "32"))


def _grid_sig(grid):
    """Identity of a parsed grid's CONTENT: the cycle's first valid time, frame
    count and file size (a re-download of the same cycle is the same data), plus
    a checksum of a sparse sample of the first frame so same-shaped but
    different data can never be mistaken for it."""
    import numpy as np
    first = np.asarray(grid["frames"][0][1])
    sample = float(np.nansum(first[:: max(1, first.size // 64)]))
    return (int(grid["frames"][0][0]), len(grid["frames"]), grid.get("size"),
            round(sample, 6))


def _points_sig(points):
    return hash(tuple((round(p[0], 6), round(p[1], 6)) for p in points))


def _memo(key, compute, pin=None):
    """Return the cached result for `key`, computing it once under a per-key
    lock. `pin` (optional) is an object the key refers to by id(); it is stored
    with the entry so the id can't be reused while the entry lives."""
    with _FIT_CACHE_LOCK:
        hit = _FIT_CACHE.get(key)
        if hit is not None and hit[0] is pin:
            return hit[1]
        lock = _FIT_KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        with _FIT_CACHE_LOCK:
            hit = _FIT_CACHE.get(key)
            if hit is not None and hit[0] is pin:
                return hit[1]
        result = compute()
        with _FIT_CACHE_LOCK:
            _FIT_CACHE.pop(key, None)
            while len(_FIT_CACHE) >= max(_FIT_CACHE_MAX, 1):
                _FIT_CACHE.pop(next(iter(_FIT_CACHE)))
            _FIT_CACHE[key] = (pin, result)
            _FIT_KEY_LOCKS.pop(key, None)
        return result


def fit_area(area, pollutant="pm25"):
    """Resolve an area's AQI once into per-lattice-point CubicSplines over the
    hourly NAQFC series, so the stream can evaluate ANY instant (smooth
    scrubbing), exactly like the NOAA observation model's per-gridpoint spline.

    `pollutant` may be a base id ("pm25", "ozone") or a NowCast variant
    ("pm25-nowcast", "ozone-nowcast"). For the raw variants each hour's
    concentration is converted straight to AQI. For NowCast, each hour's AQI is
    derived from the EPA-weighted average of the trailing NOWCAST_HOURS
    concentrations (backfilled across the prior cycle via the archive) — the
    damped number consumer apps report — THEN converted to AQI.

    Returns a dict:
      {"start": epoch, "end": epoch,      # forecast window (first/last hour)
       "points": [[lon,lat], ...],        # wire order (coords[0]=lon)
       "splines": [CubicSpline|None, ...],# parallel to points; None if no data
      }
    or None if the grid is unavailable. Splines map epoch_seconds -> AQI,
    clamped to >= 0 at evaluation (AQI can't be negative; a cubic fit can
    undershoot between knots — same guard as the NOAA non-negative attributes).
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    base, nowcast = _parse_pollutant(pollutant)
    if base not in POLLUTANTS:
        return None

    combined = _combined_series(area, base)
    if combined is None:
        return None
    epochs, series, points, _idx, grid = combined
    convert = _CONVERTER[base]

    # The result depends only on the data (cycle, grid, prior-cycle backfill)
    # and the area's points, so identical requests share ONE fit: /aqi/range,
    # each /aqi/stream fill request and the bundle builder all ask for it.
    key = ("fit", base, nowcast, _grid_sig(grid), len(epochs), int(epochs[0]),
           _points_sig(points))

    def _build():

        # The served forecast WINDOW is always the current cycle's span — prior
        # cycle hours exist only to feed NowCast's trailing average, not to extend
        # the user-visible timeline.
        win_start = int(grid["frames"][0][0])
        win_end = int(grid["frames"][-1][0])

        # Concentration -> AQI for every (hour, point) at once, NaN = no value.
        # (Per-point Python loops over ~10^4 points x ~10^2 hours were the bulk
        # of a cold fit.) NowCast first averages the trailing hours.
        conv = _CONVERTER_ARRAY[base]
        aqi_mat = conv(_nowcast_matrix(series) if nowcast else series)

        # Only fit over hours inside the current forecast window (prior-cycle
        # hours are inputs to NowCast, not knots we want to serve/extrapolate).
        in_win = (epochs >= win_start) & (epochs <= win_end)
        finite = np.isfinite(aqi_mat) & in_win[:, None]
        splines = []
        for col in range(aqi_mat.shape[1]):
            valid = finite[:, col]
            if valid.sum() < 2:
                splines.append(None)
                continue
            try:
                splines.append(CubicSpline(epochs[valid], aqi_mat[valid, col]))
            except Exception:
                splines.append(None)

        return {
            "start": win_start,
            "end": win_end,
            "points": points,
            "splines": splines,
        }

    return _memo(key, _build)


# --- Dense (spatial) fitting: stage-1 temporal + stage-2 spatial ------------

WR_AQI_UPSAMPLE = "WR_AQI_UPSAMPLE"


def fit_area_dense(area, pollutant="pm25", upsample=None):
    """Like fit_area, but SPATIALLY DENSIFIES the output lattice by `upsample`
    (WR_AQI_UPSAMPLE if None) via the shared two-stage (time-then-space)
    interpolation in lib.interp. The return shape is identical to fit_area
    ({start,end,points,splines}, points as [lon,lat]) so predict_area /
    grid_points and all consumers work UNCHANGED — just more points. At
    upsample<=1 this is a thin pass-through to fit_area (no stage-2 cost).

    AQI's per-point splines are keyed on absolute epoch and AQI is clamped >=0
    at EVALUATION (predict_area), so the dense re-fit mirrors fit_area's
    CubicSpline-over-valid-knots with no extra clamp here."""
    from . import interp

    k = interp.upsample_env(WR_AQI_UPSAMPLE) if upsample is None else max(int(upsample), 1)
    fit = fit_area(area, pollutant)
    if k <= 1 or fit is None:
        return fit
    # The (costly) spatial densify is a pure function of the shared fit + k.
    return _memo(("dense", id(fit), k), lambda: _densify(fit, k), pin=fit)


def _densify(fit, k):
    """Stage-2 spatial densify of a stage-1 fit by factor k (see fit_area_dense)."""
    import numpy as np
    from scipy.interpolate import CubicSpline
    from . import interp

    pts = fit["points"]                      # [[lon,lat], ...]
    if not pts:
        return fit
    lons = np.array([p[0] for p in pts], dtype=float)
    lats = np.array([p[1] for p in pts], dtype=float)
    epochs = _dense_epochs(fit)

    def _refit(values, ep):
        # values: dense point's interpolated series over `ep` (native knots).
        v = np.asarray(values, dtype=float)
        good = np.isfinite(v)
        if good.sum() < 2:
            return None
        try:
            return CubicSpline(ep[good], v[good])
        except Exception:
            return None

    dense = interp.densify_fit(lats, lons, epochs, fit["splines"], _refit, k)
    if dense is None:
        return fit
    d_lats, d_lons, d_splines = dense
    return {
        "start": fit["start"],
        "end": fit["end"],
        "points": [[float(lo), float(la)] for lo, la in zip(d_lons, d_lats)],
        "splines": d_splines,
    }


# Number of temporal knots at which native splines are sampled to build the
# dense value cube. The AQI fit doesn't carry its raw knot epochs through, so
# we sample the window on an hourly-ish grid; cubic splines are smooth between
# knots so this faithfully reproduces the fitted field.
_AQI_DENSE_KNOTS = 73


def _dense_epochs(fit):
    import numpy as np
    span = max(1, int(fit["end"]) - int(fit["start"]))
    n = min(_AQI_DENSE_KNOTS, max(2, span // 3600 + 1))
    return np.linspace(fit["start"], fit["end"], num=n, dtype=float)


def _vector_splines(fit):
    """A callable evaluating ALL of a fit's per-point CubicSplines at one epoch
    in a single vectorized step: returns a float array aligned to fit["points"],
    NaN where the point has no spline.

    The old per-frame loop made one scipy call per point per frame (~70 ms for a
    ~9,000-point area). `SplineBank` (lib.splinevec) stacks the coefficients once
    and evaluates with scipy's own arithmetic, so values are bit-identical. Built
    once per fit and kept on it."""
    cached = fit.get("_vec")
    if cached is not None:
        return cached
    from .splinevec import SplineBank
    evaluate = SplineBank(fit["splines"]).evaluate
    fit["_vec"] = evaluate
    return evaluate


def predict_area(fit, epochs, fixed_grid=False):
    """Evaluate a fitted area (from fit_area) at the given list of epoch
    seconds.

    Two output shapes:
      * fixed_grid=False (default, legacy): yields self-contained frames
        {"epoch", "coords":[[lon,lat],...], "vals":[...]} with per-frame
        coords (points with no spline / out of window are DROPPED). Kept for
        any caller that still wants standalone frames.
      * fixed_grid=True (grid wire format): the point set is FIXED to fit's
        points (same order every frame); yields {"epoch", "vals":[...]} where
        vals is aligned by index to fit["points"] and is `None` for a point
        whose spline is None or whose value is out of the forecast window.
        Caller emits the shared coords header once (see `grid_points`).
    AQI is clamped to >= 0 and rounded to an int."""
    import numpy as np
    pts = fit["points"]
    splines = fit["splines"]
    lo, hi = fit["start"], fit["end"]

    if fixed_grid:
        vec = _vector_splines(fit)
        none_idx = [i for i, s in enumerate(splines) if s is None]
        has_any = len(none_idx) < len(splines)
        for t in epochs:
            if not (lo <= t <= hi) or not has_any:
                continue          # whole frame is None -> no frame (as before)
            y = vec(t)            # all points at once; NaN where no spline
            y = np.where(np.isnan(y), 0.0, np.maximum(y, 0.0))   # AQI >= 0
            vals = np.round(y).astype(np.int64).tolist()         # half-even, like round()
            for i in none_idx:
                vals[i] = None
            yield {"epoch": int(t), "vals": vals}
        return

    for t in epochs:
        if t < lo or t > hi:
            continue
        co = []
        vals = []
        for p, s in zip(pts, splines):
            if s is None:
                continue
            y = float(s(t))
            if y < 0:
                y = 0.0            # non-negative clamp (AQI can't be negative)
            co.append(p)
            vals.append(round(y))
        if vals:
            yield {"epoch": int(t), "coords": co, "vals": vals}


def grid_points(fit):
    """The FIXED coords header for an AQI fit, as [[lon,lat],...] aligned to
    the vals arrays predict_area(fixed_grid=True) emits. `fit["points"]` is
    already [lon,lat] pairs."""
    return list(fit["points"])
