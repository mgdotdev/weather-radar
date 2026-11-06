"""Precomputed frame bundles for watched URLs.

A *bundle* is the entire scrubbable animation for a watched map URL, computed
ahead of time and written to disk, so that clicking the watched link reaches a
playable animation with ZERO per-frame computation or progressive streaming —
the client fetches one file and bulk-populates its in-memory frame cache.

What a bundle contains
----------------------
* ``range``  : the same payload the /type/range endpoint returns
               (min/max/uom/start/end/timeZone) — the client uses start/end to
               anchor its frame grid and min/max/uom for the color scale.
* ``frames`` : one slim record per grid tick, IDENTICAL to a series-stream
               line: ``{"time", "uom", "coords": [[c0,c1],...], "vals": [...]}``
               so the client can cache each under its existing frame key with
               no transformation.
* ``meta``   : the area + model identity (lat/lon/width/height/dt/kind/model)
               used to locate and validate the bundle.

Frame grid
----------
Frames are evaluated at the watched URL's OWN ``dt`` across the forecast window
[start, end] (the same anchor + step the client uses), so every time the client
asks for is present in the bundle — no subset/resample logic needed.

Storage & freshness
-------------------
One JSON file per (area, model, dt) under CACHE_DIR. The bundle is regenerated
only when the underlying data changed: we stamp it with the max content-change
marker mtime across the area's gridpoints (the same signal
models.utils.source_mtime uses), and treat a bundle as stale when any gridpoint
has a newer change marker. So an unchanged refresh re-stamps nothing and the
bundle is reused.
"""

import os
import json
import gzip
import hashlib
import logging
from datetime import datetime

from weather_radar.lib import framing

logger = logging.getLogger(__name__)

CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
BUNDLE_DIR = os.path.join(CACHE_ROOT, "weather-radar-bundles")

# Cap frames per bundle (mirrors the series endpoint's 5000 guard) so a
# pathological window/dt can't produce an unbounded file.
MAX_FRAMES = 5000


def _bundle_key(model, lat, lon, width, height, dt, kind):
    """Stable filename for a bundle. Rounded coords so trivially-different
    centers (float noise) map to one file; dt/kind included since they change
    the frame grid / ensemble."""
    sig = f"{model}|{kind}|{round(lat, 6)}|{round(lon, 6)}|{round(width, 1)}|{round(height, 1)}|{int(dt)}"
    h = hashlib.sha256(sig.encode()).hexdigest()[:16]
    return h


def _bundle_path(model, lat, lon, width, height, dt, kind):
    """On-disk bundle path. The bundle is GZIPPED NDJSON (.ndjson.gz): the
    first line is a {"type":"range",...} header, each subsequent line is one
    frame record {time,uom,coords,vals} — identical to a /series/stream line.
    Stored gzipped (coords repeat every frame, so it compresses ~10x) and
    streamed to the client AS-IS with Content-Encoding: gzip, so the request
    path never parses or re-serializes the (large) payload."""
    return os.path.join(
        BUNDLE_DIR,
        _bundle_key(model, lat, lon, width, height, dt, kind) + ".ndjson.gz")


def _area_source_mtime(area):
    """Newest content-change-marker mtime across the area's gridpoints — the
    freshness signal. A bundle is stale if any gridpoint changed after it was
    built. Returns 0.0 if nothing is resolved yet."""
    from weather_radar.lib.models.utils import source_mtime
    newest = 0.0
    try:
        for gp in area.iter_gridpoints():
            m = source_mtime(f"/gridpoints/{gp}")
            if m and m > newest:
                newest = m
    except Exception:
        pass
    return newest


def _aqi_grib_mtime(pollutant):
    """Freshness signal for an AQI bundle: the mtime of the NAQFC GRIB the
    bundle was built from. A bundle is stale once a newer GRIB is downloaded
    (new cycle). Returns 0.0 if unknown."""
    try:
        from weather_radar.lib import aqi
        import os as _os
        # The current grid path for this pollutant (set when ensure_grid ran).
        with aqi._CURRENT_PATH_LOCK:
            path = aqi._CURRENT_PATH.get(pollutant)
        if path and _os.path.exists(path):
            return _os.path.getmtime(path)
    except Exception:
        pass
    return 0.0


def _ndfd_grid_mtime(attribute):
    """Freshness signal for an NDFD bundle: the newest mtime of the element
    GRIB files backing `attribute`. A bundle is stale once a newer element
    file is downloaded. Returns 0.0 if unknown."""
    try:
        from weather_radar.lib import ndfd
        return ndfd.grid_mtime(attribute)
    except Exception:
        return 0.0


def _build_bundle_aqi(model, lat, lon, width, height, dt, kind):
    """Produce (header, frame_iterator, path) for an AQI bundle — the frames
    come from the NAQFC per-gridpoint CubicSpline fit (aqi.fit_area), evaluated
    over the whole forecast window at `dt`, IDENTICAL to what /aqi/stream emits
    live. `model` is the pollutant (pm25|ozone)."""
    from datetime import datetime
    from weather_radar.lib.area import MapCoordinate, CoordinateArea
    from weather_radar.lib import aqi as aqi_mod

    area = CoordinateArea(MapCoordinate(lat, lon), width, height)
    fit = aqi_mod.fit_area(area, model)
    if fit is None:
        return None
    start, end = fit["start"], fit["end"]
    step = int(dt) if dt and int(dt) > 0 else 3600
    if end <= start:
        return None

    tz = None
    try:
        tz = area.time_zone
    except Exception:
        pass
    # AQI range is the fixed display scale (mirrors the /aqi/range endpoint).
    rng = {"min": 0, "max": 300, "uom": "AQI",
           "start": datetime.fromtimestamp(start).isoformat(),
           "end": datetime.fromtimestamp(end).isoformat(), "timeZone": tz}
    header = {
        "type": "range", "range": rng,
        # Grid header: fixed coords (once) + uom, so the AQI bundle header is
        # also the stream grid header. Frames are values-only, index-aligned.
        "uom": "AQI",
        "coords": aqi_mod.grid_points(fit),
        "meta": {"model": model, "kind": kind, "lat": lat, "lon": lon,
                 "width": width, "height": height, "dt": int(dt),
                 "source_mtime": _aqi_grib_mtime(model),
                 "built": datetime.now().isoformat()},
    }

    epochs = list(range(start, end + 1, step))

    def frames():
        for fr in aqi_mod.predict_area(fit, epochs, fixed_grid=True):
            yield {"time": fr["epoch"], "vals": fr["vals"]}

    return header, frames()


def _write_bundle(path, header, frame_iter):
    """Write a gzipped BINARY-SIDECAR bundle (see weather_radar.lib.framing):
    a gzipped stream of length-prefixed records —
      * record 0 : JSON meta (bytes). A superset of the stream grid header —
                   it carries `range`/`meta` (for read_bundle_range + the
                   freshness check) AND the grid fields (uom, coords, enc,
                   scale, offset, sentinel) so serve_bundle_epochs can
                   reconstruct the stream's binary header from it.
      * record k : one binary frame [epoch][N × int16], from encode_frame.
    Atomic publish. Returns the frame count, or 0 (and no file) if none.

    `header` must contain `uom` and `coords`; `frame_iter` yields
    {"time", "vals"} dicts (vals aligned to coords, None for missing)."""
    uom = header.get("uom", "")
    coords = header.get("coords", [])
    scale, offset = framing.choose_scale(uom)
    meta_obj = dict(header)
    meta_obj.setdefault("type", "range")
    meta_obj["enc"] = framing.ENC_I16
    meta_obj["scale"] = scale
    meta_obj["offset"] = offset
    meta_obj["sentinel"] = framing.SENTINEL
    meta_bytes = json.dumps(meta_obj, separators=(",", ":")).encode("utf-8")

    tmp = path + ".tmp"
    n = 0
    try:
        os.makedirs(BUNDLE_DIR, exist_ok=True)
        with gzip.open(tmp, "wb") as f:
            f.write(framing.record(meta_bytes))
            for rec in frame_iter:
                f.write(framing.record(
                    framing.encode_frame(rec["time"], rec["vals"], scale, offset)))
                n += 1
        if n == 0:
            os.unlink(tmp)
            return 0
        os.replace(tmp, path)
    except OSError:
        logger.warning("bundle write failed for %s", path, exc_info=True)
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return 0
    return n


# Back-compat alias (callers still use the old name).
_write_bundle_ndjson = _write_bundle


def build_bundle(model, lat, lon, width, height, dt, kind):
    """Compute the full animation for a watched URL and write it to disk as
    gzipped NDJSON (header line + one {time,uom,coords,vals} frame per line).
    Returns the bundle path on success, or None.

    Dispatches on `kind`: AQI uses the NAQFC spline path (bypassing NOAA
    gridpoints); everything else uses the NOAA ensemble path. Both write the
    SAME NDJSON format, so serving is identical. Heavy — run in the background
    builder, not on a request; frames stream to disk (bounded memory)."""
    path = _bundle_path(model, lat, lon, width, height, dt, kind)

    if kind == "aqi":
        built = _build_bundle_aqi(model, lat, lon, width, height, dt, kind)
        if built is None:
            return None
        header, frames = built
        n = _write_bundle_ndjson(path, header, frames)
        if n == 0:
            return None
        logger.warning("bundle built (aqi): model=%s %dx%dkm dt=%d -> %d frames",
                       model, int(width), int(height), int(dt), n)
        return path

    # --- NOAA / NDFD ensemble path ---
    # Both sources expose the same ensemble contract (value_range + per-point
    # models with a `(start,end,center,f,units,expiry)` tuple), so the bundle
    # math below is shared. They differ only in how the per-point models are
    # enumerated: NOAA resolves WFO gridpoints (area.iter_gridpoints), NDFD
    # samples the area's lattice points from the GRIB grid (iter_points).
    from datetime import datetime
    from weather_radar.lib.area import MapCoordinate, CoordinateArea
    from weather_radar.lib.models.utils import BoundsError
    from weather_radar.server.api.models import ENSEMBLE_KINDS
    from weather_radar.lib.models import weather_source
    from weather_radar.lib.models.accumulation import AccumulationEnsemble

    is_ndfd = weather_source() == "ndfd"
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS().get(kind, AccumulationEnsemble)
    ensemble = ensemble_class(area, model)

    # Range (also yields the forecast window that anchors the frame grid).
    rng = ensemble.value_range()
    if not rng or not rng.get("start") or not rng.get("end"):
        return None
    try:
        start = int(datetime.fromisoformat(rng["start"]).timestamp())
        end = int(datetime.fromisoformat(rng["end"]).timestamp())
    except Exception:
        return None
    try:
        rng = dict(rng)
        rng["timeZone"] = area.time_zone
    except Exception:
        pass

    step = int(dt) if dt and int(dt) > 0 else 3600
    if end <= start:
        return None

    models = []
    coords = []
    uom = ""
    if is_ndfd:
        # NDFD: the ensemble fit the whole area already (value_range above);
        # iterate its wrapped lattice-point models directly — no /points.
        try:
            point_models = list(ensemble.iter_points())
        except Exception:
            point_models = []
        for m in point_models:
            try:
                _st, _et, center, _f, units, _exp = m.model
            except Exception:
                continue
            models.append(m)
            coords.append([center.lon, center.lat])  # [lon,lat]; see series_stream
            uom = uom or units
    else:
        gridpoints = list(area.iter_gridpoints())
        for gp in gridpoints:
            try:
                m = ensemble[gp]
                _st, _et, center, _f, units, _exp = m.model
            except Exception:
                continue
            models.append(m)
            coords.append([center.lon, center.lat])  # [lon,lat]; see series_stream
            uom = uom or units
    if not models:
        return None

    header = {
        "type": "range",
        "range": rng,
        # Grid header fields: the FIXED coords lattice (sent once) + uom, so a
        # bundle's header line is ALSO the stream's grid header. Frames below
        # are values-only, aligned by index to these coords.
        "uom": uom,
        "coords": coords,
        "meta": {
            "model": model, "kind": kind,
            "lat": lat, "lon": lon, "width": width, "height": height,
            "dt": int(dt),
            "source_mtime": (_ndfd_grid_mtime(model)
                             if is_ndfd else _area_source_mtime(area)),
            "built": datetime.now().isoformat(),
        },
    }

    def frames():
        t = start
        n = 0
        while t <= end + 1 and n < MAX_FRAMES:
            dtime = datetime.fromtimestamp(t)
            vals = []
            any_val = False
            for m in models:
                try:
                    y = m.predict(dtime, verbose=False)
                except BoundsError:
                    vals.append(None)   # fixed-length: null where out of horizon
                    continue
                except Exception:
                    vals.append(None)
                    continue
                vals.append(y)
                any_val = True
            if any_val:
                yield {"time": t, "vals": vals}
            t += step
            n += 1

    n_frames = _write_bundle_ndjson(path, header, frames())
    if n_frames == 0:
        return None
    logger.warning(
        "bundle built: model=%s %dx%dkm dt=%d -> %d frames (%d gridpoints)",
        model, int(width), int(height), int(dt), n_frames, len(models),
    )
    return path


def _read_bundle_meta(path):
    """Return the bundle's record-0 JSON meta dict (range/meta/uom/coords/
    enc/scale/offset/...), or None. Reads only the first length-prefixed record
    — cheap, no frame decode."""
    try:
        with gzip.open(path, "rb") as f:
            prefix = f.read(4)
            if len(prefix) < 4:
                return None
            (ln,) = framing._LEN.unpack(prefix)
            meta_bytes = f.read(ln)
            if len(meta_bytes) < ln:
                return None
            return json.loads(meta_bytes.decode("utf-8"))
    except Exception:
        return None


def _built_source_mtime(path):
    """Read the bundle's meta record and return its stored source_mtime (the
    freshness stamp), or None. Used by the builder's freshness check."""
    meta = _read_bundle_meta(path)
    if meta is None:
        return None
    return meta.get("meta", {}).get("source_mtime", 0.0)


def read_bundle_range(model, lat, lon, width, height, dt, kind):
    """Return the precomputed bundle's stored `range` (min/max/uom/start/end/
    timeZone) if a bundle keyed by these exact params exists, else None. Reads
    only the meta record — cheap, no gridpoint resolution. Used by the prefetch
    endpoint so a watched url's color range + forecast window come WITHOUT the
    usual resolve/fetch (the frames then stream from the same bundle)."""
    path = _bundle_path(model, lat, lon, width, height, dt, kind)
    if not os.path.exists(path):
        return None
    meta = _read_bundle_meta(path)
    if meta is None or meta.get("type") != "range":
        return None
    rng = meta.get("range")
    # Derive the painted-field extent from the stored lattice coords (same 5%
    # pad the client IDW raster uses), so the watched-url fast path sizes the
    # area box correctly from the first render too (the /type/range endpoint
    # adds the same `extent` on the non-bundle path). coords are [lon, lat].
    if isinstance(rng, dict) and rng.get("extent") is None:
        coords = meta.get("coords") or []
        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            pad_lon = (max_lon - min_lon) * 0.05 or 0.01
            pad_lat = (max_lat - min_lat) * 0.05 or 0.01
            rng = dict(rng)
            rng["extent"] = {
                "w": min_lon - pad_lon, "e": max_lon + pad_lon,
                "s": min_lat - pad_lat, "n": max_lat + pad_lat,
            }
    return rng


def serve_bundle_epochs(model, lat, lon, width, height, dt, kind, epochs,
                        covered, emit_header=False):
    """Yield the bundle's BINARY records for exactly the requested `epochs` (an
    iterable of int epochs). Each epoch found is added to `covered`. Yields
    nothing if no bundle exists for these params.

    BINARY SIDECAR (see weather_radar.lib.framing): when `emit_header=True`, the
    FIRST yielded record is the grid header (reconstructed from the bundle's
    meta — uom/coords/enc/scale/offset), so the client can key the stream's
    binary frames to it. Frames are the stored binary frame records, filtered
    to the wanted epochs; their encoding (scale/offset) is IDENTICAL to what the
    live path would emit (choose_scale is a pure function of uom), so a mixed
    bundle+live stream is self-consistent. The header is emitted lazily, right
    before the first matching frame, so a totally-missed request yields nothing
    (letting the live path emit its own header).

    CHEAP + bounded memory: de-frames record-by-record; NO gridpoint resolution.
    """
    path = _bundle_path(model, lat, lon, width, height, dt, kind)
    if not os.path.exists(path):
        return
    wanted = {int(e) for e in epochs}
    if not wanted:
        return
    try:
        with gzip.open(path, "rb") as f:
            # Record 0: meta. Build the binary grid header to emit lazily.
            prefix = f.read(4)
            if len(prefix) < 4:
                return
            (ln,) = framing._LEN.unpack(prefix)
            meta = json.loads(f.read(ln).decode("utf-8"))
            grid_header_rec = None
            if emit_header:
                grid_header_rec = framing.record(framing.encode_header(
                    meta.get("uom", ""), meta.get("coords", []),
                    meta.get("scale", 10.0), meta.get("offset", 0.0),
                    meta.get("sentinel", framing.SENTINEL)))

            # Records 1..N: binary frames. Peek the epoch (first 8 bytes) to
            # decide if this frame is wanted; yield the WHOLE record unchanged
            # (same encoding the live path uses).
            while True:
                prefix = f.read(4)
                if len(prefix) < 4:
                    break
                (ln,) = framing._LEN.unpack(prefix)
                payload = f.read(ln)
                if len(payload) < ln:
                    break
                try:
                    t = int(framing.decode_frame_epoch(payload))
                except Exception:
                    continue
                if t in wanted:
                    covered.add(t)
                    if grid_header_rec is not None:
                        yield grid_header_rec
                        grid_header_rec = None  # once
                    yield framing.record(payload)
    except Exception:
        logger.warning("serve_bundle_epochs failed for %s", path, exc_info=True)
        return


# --- URL -> bundle params, and the single-instance background builder --------

import threading
import time
from urllib.parse import urlparse, parse_qs

# Gate + cadence for the background builder (runs in ONE worker only — the
# leader that spawned the background processes — so bundles are built once and
# shared via disk, not rebuilt per worker).
BUILDER_ENABLED = os.environ.get("WR_BUILD_BUNDLES", "1") != "0"
BUILD_INTERVAL_SEC = float(os.environ.get("WR_BUILD_INTERVAL_SEC", "60"))
BUILD_STARTUP_DELAY_SEC = float(os.environ.get("WR_BUILD_STARTUP_DELAY_SEC", "15"))

_builder_thread = None
_builder_lock = threading.Lock()


def _params_for_url(url):
    """Resolve a watched map URL to bundle params, or None if it isn't a
    bundleable map-model URL. Mirrors the map route: model key -> (kind,
    attribute) via the server catalog; the NOAA attribute (not the dropdown
    key) is what the ensemble reads, so bundles key on it consistently with
    the live series endpoint."""
    from weather_radar.server.app import _profile, DEFAULT_PROFILE

    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "map" or parts[1] != "models":
        return None
    model_key = parts[2]
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
        return None
    width = _num("width") or 1.0
    height = _num("height") or 1.0
    dt = _num("dt")
    dt = int(dt) if dt else 3600

    profile = _profile(model_key) or {**DEFAULT_PROFILE, "attribute": model_key}
    return {
        "model": profile["attribute"],  # NOAA attribute the ensemble reads
        "kind": profile["kind"],
        "lat": lat, "lon": lon, "width": width, "height": height, "dt": dt,
    }


def _bundle_is_fresh(model, lat, lon, width, height, dt, kind):
    """Builder-side freshness check: True if an on-disk bundle exists AND its
    source hasn't changed since it was built. For NOAA kinds the source signal
    is the area's gridpoint change-markers; for AQI it's the NAQFC GRIB mtime.
    Resolves data, so this is for the BACKGROUND builder only — never a
    request path."""
    path = _bundle_path(model, lat, lon, width, height, dt, kind)
    if not os.path.exists(path):
        return False
    built = _built_source_mtime(path)
    if built is None:
        return False  # unreadable/corrupt header -> rebuild
    try:
        if kind == "aqi":
            # AQI source = the NAQFC GRIB; stale once a newer cycle downloaded.
            return _aqi_grib_mtime(model) <= built + 1e-6
        from weather_radar.lib.models import weather_source
        if weather_source() == "ndfd":
            # NDFD source = the element GRIB files; stale once a newer file is
            # downloaded. No data resolution needed (just a stat).
            return _ndfd_grid_mtime(model) <= built + 1e-6
        from weather_radar.lib.area import MapCoordinate, CoordinateArea
        area = CoordinateArea(MapCoordinate(lat, lon), width, height)
        return _area_source_mtime(area) <= built + 1e-6
    except Exception:
        # Can't evaluate -> treat as fresh so we don't rebuild in a tight loop.
        return True


def build_for_url(url):
    """Build (and persist) the bundle for one watched URL. Skips the heavy
    rebuild if the on-disk bundle is still fresh (no gridpoint changed since
    it was built), so a steady-state builder pass is cheap. Returns True if a
    bundle exists (fresh or freshly built) for the URL afterward."""
    p = _params_for_url(url)
    if p is None:
        return False
    if _bundle_is_fresh(p["model"], p["lat"], p["lon"], p["width"], p["height"],
                        p["dt"], p["kind"]):
        return True
    b = build_bundle(p["model"], p["lat"], p["lon"], p["width"], p["height"],
                     p["dt"], p["kind"])
    return b is not None


def _build_pass():
    """Build/refresh bundles for every currently-watched URL."""
    from weather_radar.lib import hotcache
    try:
        watched = hotcache.list_watched()
    except hotcache.SchedulerUnavailable:
        return
    except Exception:
        logger.warning("bundle builder: failed to list watched URLs", exc_info=True)
        return
    urls = watched.get("urls", []) if isinstance(watched, dict) else []
    for url in urls:
        try:
            build_for_url(url)
        except Exception:
            logger.warning("bundle builder: failed for %s", url, exc_info=True)


def _run():
    if BUILD_STARTUP_DELAY_SEC > 0:
        time.sleep(BUILD_STARTUP_DELAY_SEC)
    while True:
        try:
            _build_pass()
        except Exception:
            logger.warning("bundle builder: pass raised; continuing", exc_info=True)
        time.sleep(max(1.0, BUILD_INTERVAL_SEC))


def start_builder():
    """Start the single-instance background bundle builder (idempotent). Call
    ONLY from the leader worker so bundles are built once, not per worker."""
    global _builder_thread
    if not BUILDER_ENABLED:
        logger.warning("bundle builder disabled (WR_BUILD_BUNDLES=0)")
        return
    with _builder_lock:
        if _builder_thread is not None and _builder_thread.is_alive():
            return
        _builder_thread = threading.Thread(target=_run, name="bundle-builder", daemon=True)
        _builder_thread.start()
    logger.warning("bundle builder started (pid=%s interval=%.0fs)",
                   os.getpid(), BUILD_INTERVAL_SEC)
