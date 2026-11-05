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
CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
NAQFC_DIR = os.path.join(CACHE_ROOT, "weather-radar-naqfc")

# NAQFC runs at 06 and 12 UTC. We try the newest cycle first and fall back.
CYCLES = ("12", "06")

# Pollutant -> (NOMADS file infix, concentration->AQI converter key).
# `_bc` = bias-corrected (model nudged toward observations); preferred.
POLLUTANTS = {
    "pm25": "ave_1hr_pm25_bc",
    "ozone": "ave_1hr_o3_bc",
}

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


# --- Cycle discovery + download ---------------------------------------------

def _latest_cycle():
    """Return (yyyymmdd, cycle) for the newest NAQFC run that exists on NOMADS,
    trying today's 12z, today's 06z, then yesterday's. Returns None if none
    reachable."""
    import requests
    now = datetime.now(timezone.utc)
    for day_offset in (0, 1):
        day = (now - timedelta(days=day_offset)).strftime("%Y%m%d")
        for cyc in CYCLES:
            url = f"{NOMADS_BASE}/aqm.{day}/{cyc}/"
            try:
                r = requests.get(url, timeout=15)
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
    tmp = path + ".tmp"
    try:
        os.makedirs(NAQFC_DIR, exist_ok=True)
        with requests.get(url, timeout=120, stream=True) as r:
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

_GRID_CACHE = {}          # path -> {"mtime", "tree", "frames":[(epoch, values)]}
_GRID_CACHE_LOCK = threading.Lock()

# Throttle NOMADS cycle probing: remember the resolved current path per
# pollutant and only re-probe when the on-disk file is no longer fresh (its
# stamped Expires has passed). Avoids 3-6 HTTP probes on every AQI request.
_CURRENT_PATH = {}        # pollutant -> path
_CURRENT_PATH_LOCK = threading.Lock()


def _load_grid(path, pollutant):
    """Parse a NAQFC GRIB2 into (KDTree, [(epoch, values)]) and memoize it by
    (path, mtime). Re-parses when the file's mtime changes (a refresh) and
    drops the superseded entry so stale cycles don't accumulate in memory."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _GRID_CACHE_LOCK:
        cached = _GRID_CACHE.get(path)
        if cached is not None and cached["mtime"] == mtime:
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

    result = {"mtime": mtime, "tree": tree, "frames": frames}
    with _GRID_CACHE_LOCK:
        # Evict any stale cycles for OTHER pollutants' old paths is left alone;
        # just replace this path's entry (old mtime superseded).
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

    # Stale/unknown: re-probe the latest cycle and (re)download.
    cyc = _latest_cycle()
    if cyc is None:
        # NOMADS unreachable: fall back to the last file we have, even if its
        # Expires passed (stale data beats no data).
        if cur is not None and os.path.exists(cur):
            return _load_grid(cur, pollutant)
        return None
    day, c = cyc
    path = _download_grib(day, c, pollutant)
    if path is None:
        return None
    with _CURRENT_PATH_LOCK:
        _CURRENT_PATH[pollutant] = path
    return _load_grid(path, pollutant)


# --- Area sampling ----------------------------------------------------------

def fit_area(area, pollutant="pm25"):
    """Resolve an area's AQI once into per-lattice-point CubicSplines over the
    hourly NAQFC series, so the stream can evaluate ANY instant (smooth
    scrubbing), exactly like the NOAA observation model's per-gridpoint spline.

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

    grid = ensure_grid(pollutant)
    if grid is None or grid["tree"] is None or not grid["frames"]:
        return None

    coords = list(area.map_coordinates)
    if not coords:
        return None
    # KDTree built on [lat, lon]; query in that order.
    pts = np.array([[c.lat, c.lon] for c in coords])
    _, idx = grid["tree"].query(pts, k=1)   # nearest grid cell per lattice point

    convert = _CONVERTER[pollutant]
    epochs = np.array([e for e, _ in grid["frames"]], dtype=float)
    # Stack the hourly value arrays -> shape (n_hours, n_gridcells); pick the
    # columns for our lattice points -> (n_hours, n_points).
    stacked = np.vstack([v for _, v in grid["frames"]])   # (n_hours, n_cells)
    series = stacked[:, idx]                               # (n_hours, n_points)

    points = [[c.lon, c.lat] for c in coords]
    splines = []
    for col in range(series.shape[1]):
        raw = series[:, col]
        # Convert each hourly concentration -> AQI; None where out of coverage.
        aqi = np.array([convert(float(x)) for x in raw], dtype=object)
        valid = np.array([a is not None for a in aqi])
        if valid.sum() < 2:
            splines.append(None)    # need >=2 points to interpolate
            continue
        xe = epochs[valid]
        ye = np.array([a for a in aqi[valid]], dtype=float)
        # CubicSpline needs strictly increasing x; epochs are already sorted
        # ascending and unique (hourly), so pass as-is.
        try:
            splines.append(CubicSpline(xe, ye))
        except Exception:
            splines.append(None)

    return {
        "start": int(epochs[0]),
        "end": int(epochs[-1]),
        "points": points,
        "splines": splines,
    }


def predict_area(fit, epochs):
    """Evaluate a fitted area (from fit_area) at the given list of epoch
    seconds. Yields one frame dict per epoch:
      {"epoch": int, "coords": [[lon,lat],...], "vals": [aqi,...]}
    Points whose spline is None, or whose epoch is outside the forecast window,
    are dropped from that frame (coords/vals stay aligned). AQI is clamped to
    >= 0 and rounded to an int.
    """
    import numpy as np
    pts = fit["points"]
    splines = fit["splines"]
    lo, hi = fit["start"], fit["end"]
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
