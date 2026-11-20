"""Precomputed frame bundles.

A *bundle* holds animation frames that were already computed for one (area,
model), so a request for them is a disk read instead of a spline fit. It is
layered and append-only (see weather_radar.lib.bundlestore): a bundle is a
directory of immutable layer files, each holding only the frames that did not
exist when it was written.

    request 1, dt=30 min  -> layer 1: t=0, 30, 60, ...
    request 2, dt=10 min  -> layer 2: t=10, 20, 40, ...      (only what was missing)
    request 3, dt=5  min  -> layer 3: t=5, 15, 25, 35, ...
    request 4, dt=5  min  -> 0 from layer 1, 5 from layer 3, 10 from layer 2, ...

Identity
--------
``dt`` is NOT part of a bundle's identity. The key is (model, kind, lat, lon,
width, height) — so changing the time step never misses the bundle, it only
means some frames might not be stored yet.

Who writes layers
-----------------
* The background builder builds the FIRST layer for each watched URL, at that
  URL's own dt, and rebuilds it when the source data changes.
* A request that had to compute frames live saves just those frames as a new
  layer, so the next request for them is a read. (Only when at least
  MIN_LAYER_FRAMES were computed, and never beyond MAX_LAYERS layers.)
All writes go through the cache-writer process (cachewriter.write_bundle_layer);
a worker writes inline only if the writer is unreachable.

Freshness
---------
Every layer is stamped with a data-version token (``meta.source_mtime``): the
forecast-cycle id for NBM and AQI, a file mtime for NDFD, the gridpoint change
markers for the JSON source. Layers of different tokens are never mixed — the
store replaces an older generation when a newer one is written, and the reader
only serves one generation (see bundlestore.usable_layers).
"""

import os
import json
import hashlib
import logging
import threading
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from weather_radar.lib import framing
from weather_radar.lib import bundlestore

logger = logging.getLogger(__name__)

CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
BUNDLE_DIR = os.path.join(CACHE_ROOT, "weather-radar-bundles")

# Cap frames per layer (mirrors the series endpoint's 5000 guard) so a
# pathological window/dt can't produce an unbounded file.
MAX_FRAMES = 5000


def _bundle_key(model, lat, lon, width, height, kind):
    """Stable directory name for a bundle. Rounded coords so trivially-different
    centers (float noise) map to one bundle; kind is included since it changes
    the ensemble. dt is deliberately NOT included (see module docstring)."""
    sig = f"{model}|{kind}|{round(lat, 6)}|{round(lon, 6)}|{round(width, 1)}|{round(height, 1)}"
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def _bundle_dir(model, lat, lon, width, height, kind):
    """On-disk directory holding the bundle's layer files."""
    return os.path.join(BUNDLE_DIR, _bundle_key(model, lat, lon, width, height, kind))


# --- Data-version tokens ----------------------------------------------------

def _area_source_mtime(area):
    """Newest content-change-marker mtime across the area's gridpoints — the
    freshness signal for the JSON gridpoints source. Returns 0.0 if nothing is
    resolved yet."""
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


def _source_token(model, kind):
    """The CURRENT data-version token for (model, kind), cheap enough for a
    request path (no data resolution), or None if it can't be determined that
    way (the JSON gridpoints source). Same scale as the token a layer is
    stamped with, so equality means "built from the data we have now"."""
    try:
        if kind == "aqi":
            from weather_radar.lib import aqi
            return aqi.disk_cycle_token(model) or None
        from weather_radar.lib.models import weather_source
        src = weather_source()
        if src == "nbm":
            from weather_radar.lib import nbm
            return nbm.cycle_token(model) or None
        if src == "ndfd":
            from weather_radar.lib import ndfd
            return ndfd.grid_mtime(model) or None
    except Exception:
        pass
    return None


def _aqi_token_in_use(pollutant):
    """Token of the NAQFC cycle this process just loaded — read right after a
    fit to label the frames that fit produced."""
    try:
        from weather_radar.lib import aqi
        return aqi.cycle_token_in_use(pollutant) or None
    except Exception:
        return None


# --- Encoding + storing a layer ---------------------------------------------

def _store_layer(dirpath, data, token, epochs):
    """Append a layer through the cache-writer (the single writer of cache
    files); fall back to writing inline only when the writer is unreachable.
    Returns the store's status string."""
    try:
        from weather_radar.lib import cachewriter
        return cachewriter.write_bundle_layer(dirpath, data, token, epochs)
    except Exception:
        return bundlestore.save_layer(dirpath, data, token, epochs)


def _meta_object(header, scale, offset):
    meta_obj = dict(header)
    meta_obj.setdefault("type", "range")
    meta_obj["enc"] = framing.ENC_I16
    meta_obj["scale"] = scale
    meta_obj["offset"] = offset
    meta_obj["sentinel"] = framing.SENTINEL
    return meta_obj


def _write_bundle(dirpath, header, frame_iter):
    """Encode `frame_iter` ({"time", "vals"} dicts aligned to header["coords"],
    None for a missing point) into one layer and append it to the bundle at
    `dirpath`. `header` must contain `uom`, `coords` and meta.source_mtime.
    Returns the number of frames in the layer (0 if there were none)."""
    uom = header.get("uom", "")
    scale, offset = framing.choose_scale(uom)
    items = [(int(rec["time"]),
              framing.encode_frame(rec["time"], rec["vals"], scale, offset))
             for rec in frame_iter]
    if not items:
        return 0
    data, epochs = bundlestore.pack_layer(_meta_object(header, scale, offset), items)
    token = (header.get("meta") or {}).get("source_mtime")
    status = _store_layer(dirpath, data, token, epochs)
    if status not in ("ok", "dup"):
        logger.warning("bundle layer not stored (%s): %s", status, dirpath)
    return len(epochs)


def save_request_layer(model, lat, lon, width, height, kind, uom, coords,
                       items, token, rng=None):
    """Persist frames a request had to compute live as a new layer, in the
    background (the response has already been streamed).

    `items` is [(epoch, payload)] with payload = framing.encode_frame(...) — the
    SAME bytes that were streamed, so a later read returns them unchanged.
    `token` is the data-version of the data they were computed from, read right
    after the fit. Skipped when the token is unknown (the JSON source), when too
    few frames were computed to be worth a file, or on any error."""
    if token is None or len(items) < bundlestore.MIN_LAYER_FRAMES:
        return False
    try:
        scale, offset = framing.choose_scale(uom)
        header = {
            "uom": uom, "coords": coords,
            "meta": {"model": model, "kind": kind, "lat": lat, "lon": lon,
                     "width": width, "height": height,
                     "source_mtime": float(token), "source": "request",
                     "built": datetime.now().isoformat()},
        }
        if rng:
            header["range"] = rng
        data, epochs = bundlestore.pack_layer(_meta_object(header, scale, offset), items)
        dirpath = _bundle_dir(model, lat, lon, width, height, kind)
    except Exception:
        logger.warning("could not pack request layer", exc_info=True)
        return False

    def _go():
        try:
            status = _store_layer(dirpath, data, float(token), epochs)
            logger.info("bundle layer from request: %s (%d frames) -> %s",
                        os.path.basename(dirpath), len(epochs), status)
        except Exception:
            logger.warning("bundle layer save failed", exc_info=True)

    threading.Thread(target=_go, name="bundle-layer-save", daemon=True).start()
    return True


# --- Building (background builder) ------------------------------------------

def _build_bundle_aqi(model, lat, lon, width, height, kind, dt):
    """Produce (header, frame_iterator) for an AQI bundle — the frames come
    from the NAQFC per-gridpoint CubicSpline fit (aqi.fit_area_dense), evaluated
    over the whole forecast window every `dt` seconds, IDENTICAL to what
    /aqi/stream emits live. `model` is the pollutant (pm25|ozone)."""
    from weather_radar.lib.area import MapCoordinate, CoordinateArea
    from weather_radar.lib import aqi as aqi_mod

    area = CoordinateArea(MapCoordinate(lat, lon), width, height)
    fit = aqi_mod.fit_area_dense(area, model)
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
        # Grid header: fixed coords (once) + uom, so the bundle header is also
        # the stream grid header. Frames are values-only, index-aligned.
        "uom": "AQI",
        "coords": aqi_mod.grid_points(fit),
        "meta": {"model": model, "kind": kind, "lat": lat, "lon": lon,
                 "width": width, "height": height, "dt": step,
                 # Read AFTER the fit: the cycle it actually used.
                 "source_mtime": _aqi_token_in_use(model),
                 "built": datetime.now().isoformat()},
    }

    epochs = list(range(start, end + 1, step))

    def frames():
        for fr in aqi_mod.predict_area(fit, epochs, fixed_grid=True):
            yield {"time": fr["epoch"], "vals": fr["vals"]}

    return header, frames()


def build_bundle(model, lat, lon, width, height, kind, dt=3600):
    """Compute the animation for a watched URL at its own `dt` across the whole
    forecast window and append it to the bundle as a layer. Returns the bundle
    directory on success, or None.

    Dispatches on `kind`: AQI uses the NAQFC spline path; everything else uses
    the ensemble path. Heavy — run in the background builder, not on a request."""
    dirpath = _bundle_dir(model, lat, lon, width, height, kind)

    if kind == "aqi":
        built = _build_bundle_aqi(model, lat, lon, width, height, kind, dt)
        if built is None:
            return None
        header, frames = built
        n = _write_bundle(dirpath, header, frames)
        if n == 0:
            return None
        logger.warning("bundle built (aqi): model=%s %dx%dkm step=%ds -> %d frames",
                       model, int(width), int(height), header["meta"]["dt"], n)
        return dirpath

    # --- NOAA / NDFD / NBM ensemble path ---
    # All sources expose the same ensemble contract (value_range + per-point
    # models with a `(start,end,center,f,units,expiry)` tuple), so the bundle
    # math below is shared. They differ only in how the per-point models are
    # enumerated: NOAA resolves WFO gridpoints (area.iter_gridpoints), the GRIB
    # sources sample the area's lattice points from the grid (iter_points).
    from weather_radar.lib.area import MapCoordinate, CoordinateArea
    from weather_radar.lib.models.utils import BoundsError
    from weather_radar.server.api.models import ENSEMBLE_KINDS
    from weather_radar.lib.models import weather_source
    from weather_radar.lib.models.accumulation import AccumulationEnsemble

    # True for both GRIB grid sources (NDFD, NBM): iterate the ensemble's
    # lattice points (no /points resolution).
    is_grib = weather_source() in ("ndfd", "nbm")
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
    if is_grib:
        # The ensemble fit the whole area already (value_range above); iterate
        # its wrapped lattice-point models directly — no /points.
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

    # Version token read AFTER the models exist (so it describes the data they
    # were fit from).
    token = _source_token(model, kind) if is_grib else _area_source_mtime(area)
    header = {
        "type": "range",
        "range": rng,
        "uom": uom,
        "coords": coords,
        "meta": {
            "model": model, "kind": kind,
            "lat": lat, "lon": lon, "width": width, "height": height,
            "dt": step,
            "source_mtime": token,
            "built": datetime.now().isoformat(),
        },
    }

    use_epoch = hasattr(models[0], "predict_epoch")

    def frames():
        t = start
        n = 0
        while t <= end + 1 and n < MAX_FRAMES:
            dtime = None if use_epoch else datetime.fromtimestamp(t)
            vals = []
            any_val = False
            for m in models:
                try:
                    y = m.predict_epoch(t) if use_epoch else m.predict(dtime, verbose=False)
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

    n_frames = _write_bundle(dirpath, header, frames())
    if n_frames == 0:
        return None
    logger.warning(
        "bundle built: model=%s %dx%dkm step=%ds -> %d frames (%d points)",
        model, int(width), int(height), step, n_frames, len(models),
    )
    return dirpath


# --- Reading ----------------------------------------------------------------

def _first_layer_with_range(layer_list):
    for _seq, _path, meta in layer_list:
        if isinstance(meta.get("range"), dict):
            return meta
    return None


def read_bundle_range(model, lat, lon, width, height, kind):
    """Return the bundle's stored `range` (min/max/uom/start/end/timeZone/extent)
    if any current layer carries one, else None. Reads only layer headers —
    cheap, no gridpoint resolution. Used by the prefetch/range endpoints so a
    watched url's color range + forecast window come WITHOUT the usual
    resolve/fetch (the frames then stream from the same bundle)."""
    dirpath = _bundle_dir(model, lat, lon, width, height, kind)
    ls = bundlestore.usable_layers(dirpath, _source_token(model, kind))
    meta = _first_layer_with_range(ls)
    if meta is None:
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
            # Clamp to the requested area rectangle (same as _field_extent) so a
            # watched-url bundle's box matches the drawn box, not the snapped
            # gridpoint-center cloud. Reconstruct the area from bundle meta.
            try:
                from weather_radar.lib.area import MapCoordinate, CoordinateArea
                from weather_radar.lib.models import weather_source
                m = meta.get("meta") or {}
                _area = CoordinateArea(MapCoordinate(m["lat"], m["lon"]),
                                       m["width"], m["height"])
                # For `gridpoints`, clamp to the EXACT drawn rectangle so the
                # box matches what the user drew (not the lattice-snapped
                # bounds). Matches _field_extent in the api.models router.
                _gp = weather_source() == "gridpoints"
                rb = _area.requested_bounds if _gp else _area.bounds
                min_lon = max(min_lon, rb["w"]); max_lon = min(max_lon, rb["e"])
                min_lat = max(min_lat, rb["s"]); max_lat = min(max_lat, rb["n"])
            except Exception:
                _gp = False
            if _gp:
                # No pad for gridpoints: the box is sized from this extent and
                # the client re-intersects its padded raster with it, so the
                # displayed box matches the drawn rectangle exactly.
                pad_lon = 0.0 if (max_lon - min_lon) else 0.01
                pad_lat = 0.0 if (max_lat - min_lat) else 0.01
            else:
                pad_lon = (max_lon - min_lon) * 0.05 or 0.01
                pad_lat = (max_lat - min_lat) * 0.05 or 0.01
            rng = dict(rng)
            rng["extent"] = {
                "w": min_lon - pad_lon, "e": max_lon + pad_lon,
                "s": min_lat - pad_lat, "n": max_lat + pad_lat,
            }
    return rng


def serve_bundle_epochs(model, lat, lon, width, height, kind, epochs,
                        covered, emit_header=False):
    """Yield the bundle's BINARY records for exactly the requested `epochs` (an
    iterable of int epochs), in ascending time order. Each epoch found is added
    to `covered`. Yields nothing if no bundle exists for these params.

    The wanted epochs are looked up across ALL of the bundle's current layers
    and merged: with layers holding t=0,30,60 / t=10,20,40 / t=5,15,25,35 a
    dt=5 request takes 0 from layer 1, 5 from layer 3, 10 from layer 2, and so
    on — and an epoch no layer holds is simply not covered (the caller computes
    it live).

    BINARY SIDECAR (see weather_radar.lib.framing): when `emit_header=True`, the
    FIRST yielded record is the grid header (reconstructed from the first
    layer's header — uom/coords/enc/scale/offset), so the client can key the
    stream's binary frames to it. The header is emitted lazily, right before
    the first matching frame, so a totally-missed request yields nothing
    (letting the live path emit its own header). Frames are the stored binary
    records, unchanged — their encoding is identical to the live path's.

    CHEAP + bounded memory: de-frames record-by-record; NO gridpoint resolution."""
    dirpath = _bundle_dir(model, lat, lon, width, height, kind)
    wanted = {int(e) for e in epochs}
    if not wanted:
        return
    ls = bundlestore.usable_layers(dirpath, _source_token(model, kind))
    if not ls:
        return
    meta = ls[0][2]
    grid_header_rec = None
    if emit_header:
        grid_header_rec = framing.record(framing.encode_header(
            meta.get("uom", ""), meta.get("coords", []),
            meta.get("scale", 10.0), meta.get("offset", 0.0),
            meta.get("sentinel", framing.SENTINEL)))
    for epoch, payload in bundlestore.merged_frames(ls, wanted):
        covered.add(epoch)
        if grid_header_rec is not None:
            yield grid_header_rec
            grid_header_rec = None  # once
        yield framing.record(payload)


# --- URL -> bundle params, and the single-instance background builder --------

class BundleFeed:
    """Stored frames for one request, handed out in TIME order so a stream can
    interleave them with frames it computes live: send the stored frame at t=0,
    compute t=15, send the stored frame at t=30, ...

    `held` is the set of requested epochs the bundle's layers can supply (known
    from the layer headers alone, before any frame is read). `upto(limit)`
    yields the records of stored frames with epoch < limit (all of them for
    limit=None); call it with the next live epoch before computing that epoch."""

    def __init__(self, layer_list, held):
        self._layers = layer_list
        self.held = held
        self._it = bundlestore.merged_frames(layer_list, held)
        self._head = next(self._it, None)

    def header_record(self):
        """The stream's binary grid header, from the first layer's header."""
        meta = self._layers[0][2]
        return framing.record(framing.encode_header(
            meta.get("uom", ""), meta.get("coords", []),
            meta.get("scale", 10.0), meta.get("offset", 0.0),
            meta.get("sentinel", framing.SENTINEL)))

    def upto(self, limit=None):
        while self._head is not None and (limit is None or self._head[0] < limit):
            _epoch, payload = self._head
            self._head = next(self._it, None)
            yield framing.record(payload)


def open_bundle_feed(model, lat, lon, width, height, kind, epochs):
    """A BundleFeed for the requested `epochs`, or None when the bundle can't
    supply any of them (no bundle, stale generation, or no overlap)."""
    wanted = {int(e) for e in epochs}
    if not wanted:
        return None
    dirpath = _bundle_dir(model, lat, lon, width, height, kind)
    ls = bundlestore.usable_layers(dirpath, _source_token(model, kind))
    if not ls:
        return None
    held = set()
    for eps in bundlestore.plan(ls, wanted).values():
        held |= eps
    return BundleFeed(ls, held) if held else None


# --- URL -> bundle params, and the single-instance background builder --------

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
    the live series endpoint. `dt` is returned for the builder (the step the
    first layer is built at) but is not part of the bundle's identity."""
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
    dt = int(dt) if dt and dt > 0 else 3600

    profile = _profile(model_key) or {**DEFAULT_PROFILE, "attribute": model_key}
    return {
        "model": profile["attribute"],  # NOAA attribute the ensemble reads
        "kind": profile["kind"],
        "lat": lat, "lon": lon, "width": width, "height": height, "dt": dt,
    }


def _bundle_is_fresh(model, lat, lon, width, height, kind):
    """Builder-side freshness check: True if the bundle has a layer built from
    the data we have NOW. Cheap (a directory listing + a token read), but may
    resolve data for the JSON source, so this is for the BACKGROUND builder
    only — never a request path."""
    dirpath = _bundle_dir(model, lat, lon, width, height, kind)
    ls = bundlestore.layers(dirpath)
    if not ls:
        return False
    toks = [t for t in (bundlestore.token_of(m) for _, _, m in ls) if t is not None]
    try:
        if kind == "aqi":
            # Make sure this process has noticed a newly published NAQFC cycle
            # (a no-op on the fast path while the current file is fresh).
            from weather_radar.lib import aqi
            base, _nowcast = aqi._parse_pollutant(model)
            aqi.ensure_grid(base)
        token = _source_token(model, kind)
        if token is not None:
            return any(abs(t - token) < bundlestore.TOKEN_EPS for t in toks)
        # JSON gridpoints: stale once any gridpoint changed after the build.
        from weather_radar.lib.area import MapCoordinate, CoordinateArea
        area = CoordinateArea(MapCoordinate(lat, lon), width, height)
        return bool(toks) and _area_source_mtime(area) <= max(toks) + 1e-6
    except Exception:
        # Can't evaluate -> treat as fresh so we don't rebuild in a tight loop.
        return True


def build_for_url(url):
    """Build (and persist) the first layer for one watched URL. Skips the heavy
    rebuild if the bundle already has a layer from the current data, so a
    steady-state builder pass is cheap. Returns True if a bundle exists (fresh
    or freshly built) for the URL afterward."""
    p = _params_for_url(url)
    if p is None:
        return False
    if _bundle_is_fresh(p["model"], p["lat"], p["lon"], p["width"], p["height"],
                        p["kind"]):
        return True
    b = build_bundle(p["model"], p["lat"], p["lon"], p["width"], p["height"],
                     p["kind"], p["dt"])
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
        # Same pass, same process: the grid the bundle just loaded is reused
        # here, so keeping a map's pins / a watched graph warm adds no parse.
        try:
            from weather_radar.server import pointseries
            pointseries.warm_url(url)
        except Exception:
            logger.warning("series pre-warm failed for %s", url, exc_info=True)


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
