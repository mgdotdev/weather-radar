"""NDFD (National Digital Forecast Database) GRIB2 weather source.

This is a drop-in ALTERNATIVE to the `api.weather.gov` `/gridpoints` JSON
source. It produces the SAME fitted-model artifacts the JSON observation /
accumulation ensembles produce — a per-point CubicSpline over time plus the
`(start, end, center, f, units, expiry)` tuple — so the existing endpoints
(`/models/{model}/type/range`, `/type/series/stream`, `/type/ensemble`) and the
bundle builder all consume it UNCHANGED. Which source the app uses is selected
by the `WR_WEATHER_SOURCE` env var (`gridpoints` default, or `ndfd`); see
`weather_radar.lib.models.weather_ensembles`.

Why GRIB here (vs. the AQI module, which is AQI-only):
  * NDFD ships ONE national GRIB2 file PER ELEMENT (`ds.<element>.bin`), each
    holding every forecast-hour message. So "one field across all times" is a
    single download — the opposite of NBM's per-hour files. See
    docs/ndfd-roadmap.md for the full source comparison.
  * The data is the SAME underlying NWS forecast the JSON API exposes (both
    derive from the WFO digital forecast), so the two sources are
    interchangeable for the app's purposes. Caveat: the public tgftp GRIB is
    time-DECIMATED at long lead times (hourly ~36h, then 3h/6h), whereas the
    JSON retains hourly further out. The CubicSpline interpolates across the
    coarser knots, so the animation stays smooth either way.

Pipeline (mirrors lib/aqi.py)
-----------------------------
1. Download the latest NDFD cycle's element GRIB2 (both VP.001-003 and
   VP.004-007 ranges, to cover ~7 days) via the single cache-writer
   (`cachewriter.ensure_binary`) so ALL disk writes stay in that one process.
2. Parse every message once; build a scipy cKDTree over the Lambert grid's
   lat/lon arrays ONCE per file (constant across messages) for ~1us/point bulk
   nearest sampling (`codes_grib_find_nearest` is ~500ms/point, unusable).
3. For an area, nearest-sample every lattice point across all forecast hours,
   fit a per-point CubicSpline over the (epoch -> value) series, and convert
   native GRIB units to the app's display units (K->degF, m/s->mph, etc.).
4. Expose each point as an ObservationModel-SHAPED object so the ensemble is a
   structural match for the JSON `ObservationEnsemble` / `AccumulationEnsemble`.

Freshness
---------
tgftp sends `Last-Modified` + `Cache-Control: no-cache` but NO `Expires`, so
(unlike NOMADS/NAQFC) we can't stamp a future mtime for a stat-only freshness
check. NDFD republishes roughly hourly, so we use a TTL: a file younger than
`NDFD_TTL_SECONDS` (default 3600s) is fresh; older triggers a re-download
(which no-ops cheaply on tgftp's side if unchanged, but we re-stamp mtime).
"""

import os
import time
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

NDFD_BASE = os.environ.get(
    "NDFD_BASE_URL",
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd",
)
SECTOR = os.environ.get("NDFD_SECTOR", "AR.conus")
CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
NDFD_DIR = os.path.join(CACHE_ROOT, "weather-radar-ndfd")

# tgftp has no Expires; NDFD updates ~hourly. Treat a file younger than this as
# fresh (stat-only), else re-download.
NDFD_TTL_SECONDS = int(os.environ.get("NDFD_TTL_SECONDS", "3600"))

# Time-range directories to concatenate for a ~7-day horizon. VP.001-003 =
# days 1-3 (hourly early), VP.004-007 = days 4-7 (coarser). Order matters:
# earlier range first so the concatenated message series is time-ascending.
TIME_RANGES = ("VP.001-003", "VP.004-007")

# 9999 is NDFD's fill/no-data sentinel.
_FILL = 9999.0


# --- Element mapping --------------------------------------------------------
# Maps the app's kebab-case attribute -> how to read it from NDFD:
#   file      : the ds.<file>.bin element basename on tgftp.
#   units     : the app DISPLAY unit string (what the legend/readout shows),
#               AFTER the convert() below is applied.
#   convert   : native-GRIB-value -> display-value (identity if None).
#   nonneg    : clamp predictions to >= 0 (physical floor; a cubic fit can
#               undershoot between knots — same guard as the JSON path's
#               NON_NEGATIVE_ATTRIBUTES).
# Element units were VERIFIED against live files (eccodes): temp/td = K,
# wspd/wgust = m/s, rhm/sky/pop = %, qpf = kg/m^2 (==mm), snow = m, vis = m.
# Several elements decode as units="unknown" (NDFD local tables); their native
# units are hardcoded here from NWS documentation.
_K_TO_F = lambda k: (float(k) - 273.15) * 9.0 / 5.0 + 32.0
_MS_TO_MPH = lambda v: float(v) * 2.2369362920544
_MM_TO_IN = lambda v: float(v) / 25.4
_M_TO_IN = lambda v: float(v) * 39.3700787
_M_TO_MI = lambda v: float(v) / 1609.344

ELEMENTS = {
    "temperature":                   {"file": "temp",     "units": "degF", "convert": _K_TO_F,   "nonneg": False},
    "dewpoint":                      {"file": "td",       "units": "degF", "convert": _K_TO_F,   "nonneg": False},
    "apparent-temperature":          {"file": "apt",      "units": "degF", "convert": _K_TO_F,   "nonneg": False},
    "max-temperature":               {"file": "maxt",     "units": "degF", "convert": _K_TO_F,   "nonneg": False},
    "min-temperature":               {"file": "mint",     "units": "degF", "convert": _K_TO_F,   "nonneg": False},
    "relative-humidity":             {"file": "rhm",      "units": "percent", "convert": None,   "nonneg": True},
    "sky-cover":                     {"file": "sky",      "units": "percent", "convert": None,   "nonneg": True},
    "probability-of-precipitation":  {"file": "pop12",    "units": "percent", "convert": None,   "nonneg": True},
    "wind-speed":                    {"file": "wspd",     "units": "mph",  "convert": _MS_TO_MPH, "nonneg": True},
    "wind-gust":                     {"file": "wgust",    "units": "mph",  "convert": _MS_TO_MPH, "nonneg": True},
    "wind-direction":                {"file": "wdir",     "units": "degrees", "convert": None,   "nonneg": True},
    "quantitative-precipitation":    {"file": "qpf",      "units": "in",   "convert": _MM_TO_IN, "nonneg": True},
    "snowfall-amount":               {"file": "snow",     "units": "in",   "convert": _M_TO_IN,  "nonneg": True},
    "ice-accumulation":              {"file": "iceaccum", "units": "in",   "convert": _M_TO_IN,  "nonneg": True},
    "visibility":                    {"file": "vis",      "units": "mi",   "convert": _M_TO_MI,  "nonneg": True},
}


def supports(attribute):
    """True if `attribute` (kebab-case) has an NDFD element mapping."""
    return attribute in ELEMENTS


# --- Download ---------------------------------------------------------------

def _element_url(element_file, time_range):
    return f"{NDFD_BASE}/{SECTOR}/{time_range}/ds.{element_file}.bin"


def _element_path(element_file, time_range):
    # Sector + range in the name so different sectors/ranges don't collide.
    sect = SECTOR.replace(".", "")
    tr = time_range.replace(".", "")
    return os.path.join(NDFD_DIR, f"ndfd.{sect}.{tr}.{element_file}.grib2")


def _is_fresh(path):
    """True if `path` exists and is younger than the TTL (tgftp sends no
    Expires, so freshness is age-based, not Expires-stamped). A single stat."""
    try:
        return (time.time() - os.path.getmtime(path)) < NDFD_TTL_SECONDS
    except OSError:
        return False


# Negative cache for element/range combinations that don't exist on tgftp.
# NOT every element is published in every time range — e.g. ds.snow.bin and
# ds.iceaccum.bin exist for VP.001-003 (near-term) but 404 for VP.004-007
# (the extended range). Without this, every grid-cache miss re-probes the
# missing file (a 404 round-trip + a warning log line) on every request. We
# remember a 404 for NDFD_MISS_TTL_SECONDS, then allow a re-probe (in case
# NDFD later starts publishing it). Keyed by (element_file, time_range).
NDFD_MISS_TTL_SECONDS = int(os.environ.get("NDFD_MISS_TTL_SECONDS", "3600"))
_MISSING = {}             # (element_file, time_range) -> epoch when to retry
_MISSING_LOCK = threading.Lock()


def _missing_recent(element_file, time_range):
    """True if (element_file, time_range) 404'd recently and shouldn't be
    re-probed yet."""
    with _MISSING_LOCK:
        retry_at = _MISSING.get((element_file, time_range))
    return retry_at is not None and time.time() < retry_at


def _mark_missing(element_file, time_range):
    with _MISSING_LOCK:
        _MISSING[(element_file, time_range)] = time.time() + NDFD_MISS_TTL_SECONDS


def _download(element_file, time_range, force=False):
    """Ensure one NDFD element/range GRIB2 is on disk and fresh; return its
    path or None. Delegated to the cache-writer (ensure_binary) so all disk
    writes stay in that one process; inline fallback if the writer is down.

    NOTE: because tgftp sends no Expires, ensure_binary stamps mtime=now (not a
    future time). That's fine — our _is_fresh is TTL-from-mtime, so a just-
    written file reads as fresh for NDFD_TTL_SECONDS.

    A recently-404'd (element, range) is skipped (see _MISSING) so a legitimately
    absent extended-range element isn't re-probed every request."""
    path = _element_path(element_file, time_range)
    if not force and _is_fresh(path):
        return path
    # Skip re-probing a known-missing range until its retry window elapses
    # (but a stale on-disk copy, if any, is still usable).
    if not force and _missing_recent(element_file, time_range):
        return path if os.path.exists(path) else None
    url = _element_url(element_file, time_range)

    try:
        from . import cachewriter
        cachewriter.ensure_binary(url, path, force=True)  # force: TTL, not Expires
        # ensure_binary stamps mtime=Expires if present; tgftp has none, so
        # mtime stays at write time (==now) -> fresh under our TTL.
        if os.path.exists(path):
            # Guard: if the writer short-circuited on a future-Expires file
            # from some proxy, mtime could be in the future; our TTL check
            # handles that (age negative -> fresh).
            return path
        # Writer attempted but produced no file. The writer can't tell us WHY
        # (it logs the HTTP status itself), but for our purposes a persistent
        # absence == treat as missing: negative-cache so we don't re-probe
        # every request. A transient failure simply retries after the miss TTL.
        # Only do this when there's no usable stale copy already on disk.
        if not os.path.exists(path):
            _mark_missing(element_file, time_range)
            return None
    except Exception:
        pass  # writer unavailable -> inline fallback

    import requests
    os.makedirs(NDFD_DIR, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with requests.get(url, timeout=180, stream=True) as r:
            if not r.ok:
                logger.warning("NDFD download %s -> HTTP %s", url, r.status_code)
                # 404/410 = this element isn't published for this range;
                # negative-cache it. Other statuses may be transient but are
                # also cached briefly to avoid hammering.
                _mark_missing(element_file, time_range)
                return path if os.path.exists(path) else None
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(tmp, path)
        logger.warning("NDFD downloaded (inline) %s (%d bytes)",
                       os.path.basename(path), os.path.getsize(path))
        return path
    except Exception:
        logger.warning("NDFD download failed for %s", url, exc_info=True)
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return path if os.path.exists(path) else None


# --- Parsed-grid cache (per worker) -----------------------------------------
# Memoizes, per element, the KDTree over grid points + the concatenated
# per-hour (epoch, values) frames across BOTH time-range files. Keyed by the
# newest contributing file mtime so a refresh re-parses and evicts the stale
# entry. (Same scheme as aqi._GRID_CACHE.)

_GRID_CACHE = {}          # element_file -> {"mtime", "tree", "frames":[(epoch, values)]}
_GRID_CACHE_LOCK = threading.Lock()


def _parse_file(path):
    """Parse one NDFD GRIB2 into (tree_or_None, [(epoch, values_array)]).
    The tree is built from the FIRST message's lat/lon arrays (constant grid).
    Returns (tree, frames). Values arrays keep the raw GRIB values (fill=9999);
    conversion/masking happens at sample time."""
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
                vals = np.asarray(ec.codes_get_values(gid), dtype=float)
                vd = ec.codes_get(gid, "validityDate")   # yyyymmdd int
                vt = ec.codes_get(gid, "validityTime")    # HHMM int
                dt = datetime(vd // 10000, (vd // 100) % 100, vd % 100,
                              vt // 100, vt % 100, tzinfo=timezone.utc)
                frames.append((int(dt.timestamp()), vals))
            finally:
                ec.codes_release(gid)
    return tree, frames


def _load_grid(attribute):
    """Ensure the element GRIBs for `attribute` are on disk + fresh, parse and
    concatenate both time-range files, and memoize (tree, frames) per element
    keyed on the newest file mtime. Returns the parsed dict or None."""
    spec = ELEMENTS.get(attribute)
    if spec is None:
        return None
    element_file = spec["file"]

    paths = []
    for tr in TIME_RANGES:
        p = _download(element_file, tr, force=False)
        if p and os.path.exists(p):
            paths.append(p)
    if not paths:
        return None

    newest = max(os.path.getmtime(p) for p in paths)
    with _GRID_CACHE_LOCK:
        cached = _GRID_CACHE.get(element_file)
        if cached is not None and cached["mtime"] == newest:
            return cached

    tree = None
    frames = []
    for p in paths:
        try:
            t, fr = _parse_file(p)
        except Exception:
            logger.warning("NDFD parse failed for %s", p, exc_info=True)
            continue
        if tree is None:
            tree = t
        frames.extend(fr)
    if tree is None or not frames:
        return None
    # Concatenation may duplicate the boundary hour (end of VP.001-003 == start
    # of VP.004-007). Dedupe by epoch, keeping the first, and sort ascending.
    seen = set()
    deduped = []
    for epoch, vals in sorted(frames, key=lambda x: x[0]):
        if epoch in seen:
            continue
        seen.add(epoch)
        deduped.append((epoch, vals))

    result = {"mtime": newest, "tree": tree, "frames": deduped}
    with _GRID_CACHE_LOCK:
        _GRID_CACHE[element_file] = result
    logger.warning("NDFD parsed %s: %d frames (%d files)",
                   element_file, len(deduped), len(paths))
    return result


def grid_mtime(attribute):
    """Newest on-disk mtime of the element files backing `attribute`, or 0.0.
    The freshness signal for NDFD bundles (analog of aqi._aqi_grib_mtime)."""
    spec = ELEMENTS.get(attribute)
    if spec is None:
        return 0.0
    newest = 0.0
    for tr in TIME_RANGES:
        p = _element_path(spec["file"], tr)
        try:
            m = os.path.getmtime(p)
            if m > newest:
                newest = m
        except OSError:
            continue
    return newest


# --- Area fitting -----------------------------------------------------------

def fit_area(area, attribute, accumulate=False):
    """Resolve an area's NDFD field into per-lattice-point CubicSplines over
    the forecast series (epoch -> display value), so any instant can be
    evaluated (smooth scrubbing) — structurally identical to aqi.fit_area.

    Returns:
      {"start": epoch, "end": epoch,
       "points": [MapCoordinate, ...],        # lattice points with data
       "splines": [CubicSpline, ...],         # parallel; map epoch -> value
       "units": str, "nonneg": bool}
    or None if the grid/attribute is unavailable. Points whose series has < 2
    valid (non-fill) samples are dropped.

    ACCUMULATION MODE (`accumulate=True`): NDFD publishes qpf/snow/ice as
    PER-INTERVAL amounts (the fall during each forecast period), not a running
    storm total. With accumulate=True each point's series is integrated into a
    CUMULATIVE total from the forecast window start (a 0 is anchored at `start`
    so the total begins at zero), and fit with a MONOTONE interpolator
    (PchipInterpolator) so the rendered total is non-decreasing with no
    between-knot dips. Fill/no-data intervals contribute 0 to the running sum
    (no reported fall there) rather than truncating the series. Observation
    mode (the default) is unchanged — it splines the raw per-interval values.
    """
    import numpy as np
    from .area import MapCoordinate

    spec = ELEMENTS.get(attribute)
    if spec is None:
        return None
    grid = _load_grid(attribute)
    if grid is None or grid["tree"] is None or not grid["frames"]:
        return None

    coords = list(area.map_coordinates)
    if not coords:
        return None

    # KDTree built on [lat, lon]; query in that order.
    pts = np.array([[c.lat, c.lon] for c in coords])
    _, idx = grid["tree"].query(pts, k=1)

    epochs = np.array([e for e, _ in grid["frames"]], dtype=float)
    stacked = np.vstack([v for _, v in grid["frames"]])   # (n_hours, n_cells)
    series = stacked[:, idx]                               # (n_hours, n_points)

    convert = spec["convert"]
    start_epoch = int(epochs[0])
    out_points = []
    out_splines = []
    for col in range(series.shape[1]):
        spline = _fit_column(series[:, col], epochs, start_epoch, convert,
                             accumulate)
        if spline is None:
            continue
        out_points.append(MapCoordinate(lat=coords[col].lat, lon=coords[col].lon))
        out_splines.append(spline)

    if not out_splines:
        return None

    return {
        "start": start_epoch,
        "end": int(epochs[-1]),
        "points": out_points,
        "splines": out_splines,
        "units": spec["units"],
        "nonneg": spec["nonneg"],
    }


# --- Per-point temporal fit (stage 1) --------------------------------------
# Factored out of fit_area so fit_area_dense can reuse the EXACT same temporal
# fitting (handling of fill/no-data, non-uniform time knots, and the monotone
# accumulation integration) on both the native lattice and the densified one.

def _fit_column(raw, epochs, start_epoch, convert, accumulate):
    """Fit ONE point's time series (raw values over `epochs`) to an
    interpolator mapping absolute-epoch -> display value, or None if the series
    can't be fit (too few valid samples). Mirrors the original fit_area loop:
    observation mode splines the raw per-interval values (CubicSpline);
    accumulation mode integrates a cumulative, monotone total (PchipInterpolator).
    `raw` may contain _FILL sentinels, which are masked (obs) or treated as 0
    contribution (accumulate)."""
    import numpy as np
    from scipy.interpolate import CubicSpline, PchipInterpolator

    raw = np.asarray(raw, dtype=float)
    valid = raw < (_FILL - 1.0)       # mask fill/no-data

    if accumulate:
        # Cumulative total from the window start. Fill intervals contribute
        # 0 (no reported fall) so the running sum stays defined and monotonic
        # across the whole window. Need at least one real sample.
        if valid.sum() < 1:
            return None
        amounts = np.where(valid, raw, 0.0).astype(float)
        if convert is not None:
            amounts = np.array([convert(v) for v in amounts], dtype=float)
        # Per-interval amounts are non-negative; guard any convert noise.
        amounts = np.clip(amounts, 0.0, None)
        cumulative = np.cumsum(amounts)
        # Anchor a 0 at the window start so the total begins at zero and the
        # first segment interpolates 0 -> first-interval total monotonically.
        xe = np.concatenate(([float(start_epoch)], epochs))
        yv = np.concatenate(([0.0], cumulative))
        # Drop any duplicate x (e.g. first frame exactly at start) so the
        # interpolator gets strictly increasing knots.
        keep = np.concatenate(([True], np.diff(xe) > 0))
        xe, yv = xe[keep], yv[keep]
        if xe.size < 2:
            return None
        try:
            # PCHIP: shape-preserving, monotone on monotone data -> the
            # cumulative total never dips between knots (CubicSpline can).
            return PchipInterpolator(xe, yv)
        except Exception:
            return None

    if valid.sum() < 2:
        return None
    xe = epochs[valid]
    yv = raw[valid]
    if convert is not None:
        yv = np.array([convert(v) for v in yv], dtype=float)
    else:
        yv = yv.astype(float)
    try:
        return CubicSpline(xe, yv)
    except Exception:
        return None


# --- Dense area fitting (stage 1 time + stage 2 space) ----------------------

def _upsample_factor():
    """Spatial upsample factor from WR_NDFD_UPSAMPLE (default 1 = off). Each
    native lattice cell becomes a (k x k) finer lattice. Clamped to >= 1."""
    try:
        k = int(os.environ.get("WR_NDFD_UPSAMPLE", "1"))
    except (TypeError, ValueError):
        return 1
    return max(k, 1)


def fit_area_dense(area, attribute, accumulate=False, upsample=None):
    """Like fit_area, but SPATIALLY DENSIFIES the output lattice by `upsample`
    (WR_NDFD_UPSAMPLE if None) via a two-stage separable interpolation:

      stage 1 (time)  : fit the native lattice's per-point temporal splines
                        exactly as fit_area does (handles fill/no-data and
                        non-uniform time knots per point).
      stage 2 (space) : at each native time knot, evaluate those splines onto
                        the native (lat x lon) grid and cubic-interpolate that
                        grid onto a finer lattice; this gives every DENSE point
                        a value series over the native knots, which is then fit
                        with a temporal interpolator (same _fit_column path).

    Returns the SAME dict shape as fit_area ({start,end,points,splines,units,
    nonneg}) so every consumer (the ensemble, the bundle builder, the client)
    works UNCHANGED — there are simply more points. At upsample<=1 this is a
    thin pass-through to fit_area (identical output, no stage-2 cost).

    The native lattice is a regular grid (area.map_coordinates emits i*step,
    j*step for every (i,j)); the axes are recovered from the sampled points so
    this does not depend on area.index_bounds (keeps the test stub simple)."""
    import numpy as np

    k = _upsample_factor() if upsample is None else max(int(upsample), 1)
    if k <= 1:
        return fit_area(area, attribute, accumulate=accumulate)

    spec = ELEMENTS.get(attribute)
    if spec is None:
        return None
    grid = _load_grid(attribute)
    if grid is None or grid["tree"] is None or not grid["frames"]:
        return None

    coords = list(area.map_coordinates)
    if not coords:
        return None

    # Native regular-lattice axes, recovered from the sampled coordinates.
    lat_ax = np.unique(np.array([c.lat for c in coords], dtype=float))
    lon_ax = np.unique(np.array([c.lon for c in coords], dtype=float))
    n_lat, n_lon = lat_ax.size, lon_ax.size
    # Need at least a 2x2 native grid to interpolate spatially; otherwise fall
    # back to the plain (undensified) fit.
    if n_lat < 2 or n_lon < 2:
        return fit_area(area, attribute, accumulate=accumulate)

    # Nearest-sample every native cell across all forecast hours.
    gy, gx = np.meshgrid(lat_ax, lon_ax, indexing="ij")   # (n_lat, n_lon)
    flat = np.column_stack([gy.ravel(), gx.ravel()])
    _, idx = grid["tree"].query(flat, k=1)

    epochs = np.array([e for e, _ in grid["frames"]], dtype=float)
    stacked = np.vstack([v for _, v in grid["frames"]])   # (n_hours, n_cells)
    native = stacked[:, idx]                              # (n_hours, n_lat*n_lon)
    n_hours = native.shape[0]
    convert = spec["convert"]
    start_epoch = int(epochs[0])

    # Dense output axes: k samples per native interval (endpoint-inclusive).
    dense_lat = _densify_axis(lat_ax, k)
    dense_lon = _densify_axis(lon_ax, k)
    dgy, dgx = np.meshgrid(dense_lat, dense_lon, indexing="ij")
    dense_query = np.column_stack([dgy.ravel(), dgx.ravel()])
    n_dense = dense_query.shape[0]

    # Stage 2 per native knot: build a (n_lat, n_lon) value grid and resample
    # onto the dense lattice. Fill/no-data cells are neighbor-filled FIRST so
    # the grid handed to the interpolator is complete (holes aren't allowed on
    # a regular-grid interpolator). dense_series[h] is the dense values at hour h.
    dense_series = np.empty((n_hours, n_dense), dtype=float)
    for h in range(n_hours):
        g = native[h].reshape(n_lat, n_lon).astype(float)
        g = _fill_holes(g)                                # complete the grid
        dense_series[h] = _resample_grid(lat_ax, lon_ax, g, dense_query)

    # Stage 1 (re-fit) on the dense points: each dense column is now a complete
    # time series over the native knots. _FILL may survive only if an entire
    # native grid at some hour was fill (then _fill_holes left it); _fit_column
    # masks those per point, so the dense fit degrades gracefully.
    from .area import MapCoordinate
    out_points = []
    out_splines = []
    for col in range(n_dense):
        spline = _fit_column(dense_series[:, col], epochs, start_epoch, convert,
                             accumulate)
        if spline is None:
            continue
        out_points.append(MapCoordinate(lat=float(dense_query[col, 0]),
                                        lon=float(dense_query[col, 1])))
        out_splines.append(spline)

    if not out_splines:
        return None

    return {
        "start": start_epoch,
        "end": int(epochs[-1]),
        "points": out_points,
        "splines": out_splines,
        "units": spec["units"],
        "nonneg": spec["nonneg"],
    }


def _densify_axis(axis, k):
    """Return a finer 1-D axis with k sub-samples per native interval,
    endpoint-inclusive and strictly increasing. k=1 returns the axis unchanged."""
    import numpy as np
    if k <= 1 or axis.size < 2:
        return axis
    segs = [np.linspace(axis[i], axis[i + 1], k + 1)[:-1]
            for i in range(axis.size - 1)]
    segs.append(axis[-1:])            # include the final endpoint once
    return np.concatenate(segs)


def _fill_holes(grid):
    """Replace _FILL/no-data cells in a 2-D grid with nearest valid values so
    the grid is complete for spatial interpolation. If EVERY cell is fill the
    grid is returned unchanged (the temporal fit masks it downstream)."""
    import numpy as np
    g = np.array(grid, dtype=float)
    bad = g >= (_FILL - 1.0)
    if not bad.any():
        return g
    if bad.all():
        return g
    try:
        from scipy.ndimage import distance_transform_edt
        # Index of the nearest valid cell for every (bad) cell.
        _, (ii, jj) = distance_transform_edt(bad, return_indices=True)
        g[bad] = g[ii[bad], jj[bad]]
    except Exception:
        # Fallback: fill with the mean of valid cells.
        g[bad] = float(g[~bad].mean())
    return g


def _resample_grid(lat_ax, lon_ax, grid, query):
    """Cubic-resample a complete (lat x lon) grid onto `query` [[lat,lon],...].
    Falls back to linear if this scipy lacks N-D cubic for RegularGridInterpolator."""
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    try:
        rgi = RegularGridInterpolator((lat_ax, lon_ax), grid, method="cubic",
                                      bounds_error=False, fill_value=None)
        return rgi(query)
    except (ValueError, NotImplementedError):
        if not getattr(_resample_grid, "_warned", False):
            logger.warning("NDFD dense: cubic RegularGridInterpolator "
                           "unavailable; falling back to linear")
            _resample_grid._warned = True
        rgi = RegularGridInterpolator((lat_ax, lon_ax), grid, method="linear",
                                      bounds_error=False, fill_value=None)
        return rgi(query)
