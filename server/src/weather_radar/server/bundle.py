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


def build_bundle(model, lat, lon, width, height, dt, kind):
    """Compute the full animation for a watched URL and write it to disk as
    gzipped NDJSON. Returns the bundle path on success, or None if there's no
    data to bundle.

    Heavy: evaluates every gridpoint at every frame. Intended to run in the
    background (the bundle builder), NOT on a request. Frames are written
    INCREMENTALLY to the gzip stream as they're computed, so peak memory stays
    bounded regardless of area size."""
    # Imported here to avoid pulling the model/ensemble stack into modules that
    # only read bundles.
    from datetime import datetime
    from weather_radar.lib.area import MapCoordinate, CoordinateArea
    from weather_radar.lib.models.utils import BoundsError
    from weather_radar.server.api.models import ENSEMBLE_KINDS
    from weather_radar.lib.models.accumulation import AccumulationEnsemble

    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
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
    # Mirror the client: timezone comes from the area center.
    try:
        rng = dict(rng)
        rng["timeZone"] = area.time_zone
    except Exception:
        pass

    step = int(dt) if dt and int(dt) > 0 else 3600
    if end <= start:
        return None

    # Precompute each gridpoint's coord once (as the stream endpoint does:
    # [center.lat, center.lon]). Skip models that fail to build.
    gridpoints = list(area.iter_gridpoints())
    models = []
    coords = []
    uom = ""
    for gp in gridpoints:
        try:
            m = ensemble[gp]
            _st, _et, center, _f, units, _exp = m.model
        except Exception:
            continue
        models.append(m)
        coords.append([center.lat, center.lon])
        uom = uom or units
    if not models:
        return None

    source_mtime = _area_source_mtime(area)
    header = {
        "type": "range",
        "range": rng,
        "meta": {
            "model": model, "kind": kind,
            "lat": lat, "lon": lon, "width": width, "height": height,
            "dt": int(dt), "source_mtime": source_mtime,
            "built": datetime.now().isoformat(),
        },
    }

    # Write gzipped NDJSON: header line, then one frame record per line,
    # streamed as each frame is computed (bounded memory). Atomic publish via
    # a temp file + os.replace so a reader never sees a partial bundle.
    path = _bundle_path(model, lat, lon, width, height, dt, kind)
    tmp = path + ".tmp"
    n_frames = 0
    try:
        os.makedirs(BUNDLE_DIR, exist_ok=True)
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(json.dumps(header, separators=(",", ":")) + "\n")
            t = start
            n = 0
            while t <= end + 1 and n < MAX_FRAMES:
                dtime = datetime.fromtimestamp(t)
                fcoords = []
                vals = []
                for m, c in zip(models, coords):
                    try:
                        y = m.predict(dtime, verbose=False)
                    except BoundsError:
                        continue
                    except Exception:
                        continue
                    fcoords.append(c)
                    vals.append(y)
                if vals:
                    rec = {"time": t, "uom": uom, "coords": fcoords, "vals": vals}
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    n_frames += 1
                t += step
                n += 1
        if n_frames == 0:
            os.unlink(tmp)
            return None
        os.replace(tmp, path)
    except OSError:
        logger.warning("bundle write failed for %s", path, exc_info=True)
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return None

    logger.warning(
        "bundle built: model=%s %dx%dkm dt=%d -> %d frames (%d gridpoints)",
        model, int(width), int(height), int(dt), n_frames, len(models),
    )
    return path


def _built_source_mtime(path):
    """Read just the header line of a gzipped bundle and return its stored
    source_mtime (the freshness stamp), or None. Used by the builder's
    freshness check — reads one line, not the whole file."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            header = json.loads(f.readline())
        return header.get("meta", {}).get("source_mtime", 0.0)
    except Exception:
        return None


def read_bundle_range(model, lat, lon, width, height, dt, kind):
    """Return the precomputed bundle's stored `range` (min/max/uom/start/end/
    timeZone) if a bundle keyed by these exact params exists, else None. Reads
    only the header line — cheap, no gridpoint resolution. Used by the prefetch
    endpoint so a watched url's color range + forecast window come WITHOUT the
    usual resolve/fetch (the frames then stream from the same bundle)."""
    path = _bundle_path(model, lat, lon, width, height, dt, kind)
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            header = json.loads(f.readline())
    except Exception:
        return None
    if header.get("type") != "range":
        return None
    return header.get("range")


def serve_bundle_epochs(model, lat, lon, width, height, dt, kind, epochs, covered):
    """Yield the bundle's frame lines for exactly the requested `epochs` (an
    iterable of int epochs), skipping the range header. Each epoch actually
    found is added to the `covered` set (so the caller can live-compute only
    the remainder). Yields nothing if no bundle exists for these params.

    This serves BOTH the sequence and the batch of a /series/stream request —
    the bundle has the whole window, so a watched url is satisfied almost
    entirely from disk; only epochs the bundle happens to lack fall through to
    live compute. Lines are newline-terminated and byte-compatible with
    live-computed frames.

    CHEAP + bounded memory: decompresses line-by-line; never loads the whole
    bundle; NO gridpoint resolution.
    """
    path = _bundle_path(model, lat, lon, width, height, dt, kind)
    if not os.path.exists(path):
        return
    wanted = {int(e) for e in epochs}
    if not wanted:
        return
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = True
        for line in f:
            if not line.strip():
                continue
            if first:
                first = False
                try:
                    if json.loads(line).get("type") == "range":
                        continue  # skip header
                except Exception:
                    pass
            try:
                t = int(json.loads(line).get("time"))
            except Exception:
                continue
            if t in wanted:
                covered.add(t)
                yield line if line.endswith("\n") else line + "\n"


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
    """Builder-side freshness check: True if an on-disk bundle exists AND no
    gridpoint's data has changed since it was built. Resolves gridpoints (via
    _area_source_mtime), so this is for the BACKGROUND builder only — never the
    request path."""
    path = _bundle_path(model, lat, lon, width, height, dt, kind)
    if not os.path.exists(path):
        return False
    built = _built_source_mtime(path)
    if built is None:
        return False  # unreadable/corrupt header -> rebuild
    try:
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
