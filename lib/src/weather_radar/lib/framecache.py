"""Server-side cache for computed ensemble frames.

The `/type/ensemble` endpoint computes a GeoJSON FeatureCollection for one
area at one instant by evaluating every gridpoint's fitted spline. During
animation the client requests the same frames over and over (looping, or
revisiting a time), and each repeat re-runs that whole spline evaluation.

This module caches the *computed* FeatureCollection per frame so a repeated
request for the same `(area, model, kind, time)` is a cheap disk read instead
of a recompute. Keying is PER FRAME (the timestamp is part of the key, but the
time-step `dt` is NOT), so a frame computed while stepping at one dt is reused
by any other dt whose grid lands on the same instant.

Design notes / invariants:
  * Frame files are DERIVED artifacts (computed from the already-cached NOAA
    gridpoint data), not raw NOAA responses. The "cachewriter owns all NOAA
    writes" rule concerns the NOAA source cache; these frame artifacts are a
    separate, read-mostly layer. Writes are atomic (temp -> os.replace) so a
    half-written frame is never observed by a concurrent reader.
  * Freshness is defined by the inputs: a frame is valid only while every
    `/gridpoints/{gp}` path for its area is still fresh. All frames in an area
    depend on the same gridpoints, so freshness is checked once per request at
    the area level. When NOAA data expires and is refreshed, stale frames are
    ignored and recomputed.
"""

import hashlib
import json
import os
import tempfile

from .cachewriter import CACHE_ROOT, is_fresh


FRAMES_DIR = os.path.join(CACHE_ROOT, "frames")


def area_key(model, kind, lat, lon, width, height):
    """Stable id for an area+model+kind — everything that determines a frame
    except the timestamp. Coordinates/sizes are rounded so trivially different
    floats (e.g. 47.6800001) collapse to the same area."""
    parts = [
        str(model),
        str(kind),
        f"{float(lat):.4f}",
        f"{float(lon):.4f}",
        f"{float(width):.3f}",
        f"{float(height):.3f}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _area_dir(akey):
    return os.path.join(FRAMES_DIR, akey)


def frame_path(akey, time):
    """Disk path for one frame of an area at `time` (epoch seconds)."""
    return os.path.join(_area_dir(akey), f"{int(time)}.json")


def area_is_fresh(grid_paths):
    """True if every `/gridpoints/{gp}` path for the area is fresh. Gates the
    whole area's cached frames at once (all share the same gridpoints)."""
    return all(is_fresh(gp) for gp in grid_paths)


def data_version(grid_paths):
    """A fingerprint of the CURRENT underlying NOAA gridpoint data for an area,
    derived from each gridpoint file's stored Expires. When NOAA data is
    refreshed the Expires changes, so the version changes — which is how a
    frame computed from old data is detected as stale even though the (new)
    gridpoints are themselves fresh. A missing/undated gridpoint contributes a
    sentinel so the version still changes once it's fetched."""
    from .cachewriter import _tempfile_for
    h = hashlib.sha256()
    for gp in sorted(grid_paths):
        _url, _d, cache_file = _tempfile_for(gp)
        token = "missing"
        try:
            with open(cache_file, "r") as f:
                body = json.load(f)
            token = body["headers"]["Expires"]
        except Exception:
            pass
        h.update(gp.encode("utf-8"))
        h.update(b"\x00")
        h.update(token.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def read_frame(akey, time, version=None):
    """Return the cached FeatureCollection for a frame, or None on a miss,
    unreadable file, or — when `version` is given — a data-version mismatch
    (the frame was computed from now-superseded NOAA data, so it's stale)."""
    path = frame_path(akey, time)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            stored = json.load(f)
    except Exception:
        return None
    if version is not None and stored.get("_version") != version:
        return None  # built from older data -> invalid
    return stored.get("fc")


def write_frame(akey, time, fc, version=None):
    """Atomically write one frame's FeatureCollection to the cache, stamped
    with the data `version` it was computed from so a later read can detect
    staleness when the underlying NOAA data has changed."""
    d = _area_dir(akey)
    os.makedirs(d, exist_ok=True)
    record = {"_version": version, "fc": fc}
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f)
        os.replace(tmp, frame_path(akey, time))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def clear_area(akey):
    """Delete every cached frame for an area. Optional belt-and-suspenders
    cleanup (the data_version check already prevents serving stale frames);
    useful to reclaim disk after a refresh. Returns the count removed."""
    d = _area_dir(akey)
    if not os.path.isdir(d):
        return 0
    removed = 0
    for name in os.listdir(d):
        if name.endswith(".json") or name.endswith(".tmp"):
            try:
                os.remove(os.path.join(d, name))
                removed += 1
            except OSError:
                pass
    return removed

