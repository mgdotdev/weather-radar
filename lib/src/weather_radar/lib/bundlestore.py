"""Layered, append-only store for precomputed animation bundles.

A *bundle* is the set of frames already computed for one (area, model). It is
stored as a DIRECTORY of immutable LAYER files::

    <bundle dir>/00001.bin.gz   <- e.g. built for dt=30 min: t=0, 30, 60, ...
    <bundle dir>/00002.bin.gz   <- a later dt=10 request:    t=10, 20, 40, ...
    <bundle dir>/00003.bin.gz   <- a later dt=5  request:    t=5, 15, 25, 35, ...

Each layer holds ONLY the frames that did not exist yet when it was written, so
no file is ever rewritten. A reader works out which layer holds each wanted
epoch (from the epoch list in every layer's header) and merges the layers back
into one time-ordered stream: a dt=5 request takes 0 from layer 1, 5 from layer
3, 10 from layer 2, and so on. Layer files are gzipped length-prefixed records:
record 0 is a JSON header (uom/coords/enc/scale/offset/sentinel, the optional
color ``range`` and a ``meta`` block, plus ``epochs``), records 1..N are binary
frames [epoch][N x int16] in ASCENDING epoch order (see weather_radar.lib.framing).

Freshness: every layer carries ``meta.source_mtime``, a *data-version token*
(a forecast-cycle id, or a file mtime for NDFD) that only ever increases when
the source data changes. Layers of different tokens are never mixed: saving a
layer with a NEWER token deletes the older layers, and a layer with an older
token than what is stored is refused. The reader serves the layers whose token
matches the current one, falling back to the newest stored generation (for a
bounded time) when the data has just refreshed and the builder hasn't caught up.

All WRITES here (``save_layer``) are meant to run in the cache-writer process
(cachewriter.write_bundle_layer); a worker only calls it directly as a fallback
when the writer is unreachable. Slot allocation is race-safe across processes
(an exclusive hard-link claims the sequence number)."""

import gzip
import heapq
import json
import logging
import os
import re
import threading
import time

from . import framing

logger = logging.getLogger(__name__)

CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
BUNDLE_ROOT = os.path.join(CACHE_ROOT, "weather-radar-bundles")

# Most layers one bundle may accumulate. Every layer is another file to open on
# a read, so this bounds the read cost; further saves are refused ("full").
MAX_LAYERS = int(os.environ.get("WR_BUNDLE_MAX_LAYERS", "16"))
# A request only writes a layer when it computed at least this many new frames
# (tiny scrub/gap fills would otherwise create a file apiece).
MIN_LAYER_FRAMES = int(os.environ.get("WR_BUNDLE_LAYER_MIN_FRAMES", "12"))
# When the data has refreshed but no layer with the new token exists yet, the
# previous generation is still served for this long (the builder normally
# replaces it within a minute or so).
STALE_FALLBACK_SECONDS = float(os.environ.get("WR_BUNDLE_STALE_SECONDS", "7200"))

TOKEN_EPS = 1e-3

_LAYER_RE = re.compile(r"^(\d{5,})\.bin\.gz$")
_SAVE_LOCK = threading.Lock()

# Layers are immutable, so a parsed header is valid for the file's lifetime.
_META_CACHE = {}
_META_CACHE_LOCK = threading.Lock()
_META_CACHE_MAX = 512


# --- Reading ----------------------------------------------------------------

def layer_paths(dirpath):
    """[(seq, path)] of the layer files in `dirpath`, ascending by seq."""
    try:
        names = os.listdir(dirpath)
    except OSError:
        return []
    out = []
    for n in names:
        m = _LAYER_RE.match(n)
        if m:
            out.append((int(m.group(1)), os.path.join(dirpath, n)))
    out.sort()
    return out


def read_meta(path):
    """The layer's record-0 JSON header (cached), or None if unreadable.
    Reads only the first record — no frame decode."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_size, st.st_mtime_ns)
    with _META_CACHE_LOCK:
        hit = _META_CACHE.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
    try:
        with gzip.open(path, "rb") as f:
            prefix = f.read(4)
            if len(prefix) < 4:
                return None
            (ln,) = framing._LEN.unpack(prefix)
            raw = f.read(ln)
            if len(raw) < ln:
                return None
            meta = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    with _META_CACHE_LOCK:
        if len(_META_CACHE) >= _META_CACHE_MAX:
            _META_CACHE.clear()
        _META_CACHE[path] = (key, meta)
    return meta


def token_of(meta):
    """The layer's data-version token (None if absent)."""
    try:
        v = meta.get("meta", {}).get("source_mtime")
        return None if v is None else float(v)
    except (TypeError, ValueError, AttributeError):
        return None


def layers(dirpath):
    """[(seq, path, meta)] for every readable layer, ascending by seq."""
    out = []
    for seq, path in layer_paths(dirpath):
        meta = read_meta(path)
        if meta is not None and isinstance(meta.get("epochs"), list):
            out.append((seq, path, meta))
    return out


def _same(a, b):
    return a is not None and b is not None and abs(a - b) < TOKEN_EPS


def usable_layers(dirpath, current_token=None, now=None,
                  stale_seconds=None):
    """The layers to read from, as [(seq, path, meta)] in write order.

    One DATA GENERATION only (frames from different forecast cycles must not be
    interleaved):
      * layers whose token equals `current_token`, if any exist; else
      * the newest stored generation — but, when `current_token` is known (so we
        know this generation is out of date), only while it is younger than
        `stale_seconds`, bounding how long a never-refreshed bundle is served;
      * with `current_token` None (a source whose version can't be checked
        cheaply) the newest generation is used as-is.
    All kept layers must share the first layer's point count and units."""
    ls = layers(dirpath)
    if not ls:
        return []
    toks = [token_of(m) for _, _, m in ls]
    chosen = []
    if current_token is not None:
        chosen = [l for l, t in zip(ls, toks) if _same(t, current_token)]
    if not chosen:
        known = [t for t in toks if t is not None]
        newest = max(known) if known else None
        chosen = [l for l, t in zip(ls, toks)
                  if (t is None and newest is None) or _same(t, newest)]
        if current_token is not None and chosen:
            now = time.time() if now is None else now
            limit = STALE_FALLBACK_SECONDS if stale_seconds is None else stale_seconds
            built = max(float(m.get("meta", {}).get("built_ts") or 0.0)
                        for _, _, m in chosen)
            if now - built > limit:
                return []
    if not chosen:
        return []
    first = chosen[0][2]
    n = len(first.get("coords") or [])
    uom = first.get("uom")
    return [l for l in chosen
            if len(l[2].get("coords") or []) == n and l[2].get("uom") == uom]


def iter_frames(path, wanted):
    """Yield (epoch, payload) for the `wanted` epochs stored in the layer, in
    ascending order. A gzip stream can't be seeked, so this reads forward and
    STOPS as soon as every wanted epoch is found or the last one is passed."""
    wanted = set(wanted)
    if not wanted:
        return
    last = max(wanted)
    found = 0
    try:
        with gzip.open(path, "rb") as f:
            prefix = f.read(4)
            if len(prefix) < 4:
                return
            (ln,) = framing._LEN.unpack(prefix)
            f.read(ln)                                  # skip the header
            while True:
                prefix = f.read(4)
                if len(prefix) < 4:
                    return
                (ln,) = framing._LEN.unpack(prefix)
                payload = f.read(ln)
                if len(payload) < ln:
                    return
                try:
                    t = int(framing.decode_frame_epoch(payload))
                except Exception:
                    continue
                if t in wanted:
                    found += 1
                    yield t, payload
                if found >= len(wanted) or t >= last:
                    return
    except Exception:
        logger.warning("bundle layer read failed: %s", path, exc_info=True)
        return


def plan(layer_list, wanted):
    """Assign each wanted epoch to the FIRST layer (write order) that holds it.
    Returns {path: set(epochs)} for layers that supply something."""
    remaining = set(int(e) for e in wanted)
    out = {}
    for _seq, path, meta in layer_list:
        if not remaining:
            break
        have = remaining.intersection(int(e) for e in meta["epochs"])
        if have:
            out[path] = have
            remaining -= have
    return out


def merged_frames(layer_list, wanted):
    """Yield (epoch, payload) for every wanted epoch the layers hold, merged
    into ONE ascending-time stream across all contributing layers."""
    assigned = plan(layer_list, wanted)
    its = [iter_frames(path, eps) for path, eps in assigned.items()]
    if not its:
        return
    if len(its) == 1:
        yield from its[0]
        return
    yield from heapq.merge(*its, key=lambda x: x[0])


# --- Writing ----------------------------------------------------------------

def pack_layer(meta_obj, items):
    """Build a layer file's bytes. `meta_obj` is the header dict (uom, coords,
    enc/scale/offset/sentinel, meta, optional range); `items` is an iterable of
    (epoch, payload) with payload = framing.encode_frame(...). Frames are sorted
    by epoch and de-duplicated (the header's `epochs` and the file agree).
    Returns (gzipped bytes, [epochs])."""
    by_epoch = {}
    for epoch, payload in items:
        by_epoch[int(epoch)] = payload
    epochs = sorted(by_epoch)
    meta = dict(meta_obj)
    meta["epochs"] = epochs
    meta.setdefault("type", "range")
    m = dict(meta.get("meta") or {})
    m.setdefault("built_ts", time.time())
    meta["meta"] = m
    blob = bytearray(framing.record(
        json.dumps(meta, separators=(",", ":")).encode("utf-8")))
    for e in epochs:
        blob += framing.record(by_epoch[e])
    return gzip.compress(bytes(blob), compresslevel=6, mtime=0), epochs


def _delete_layers(ls):
    for _seq, path, _meta in ls:
        try:
            os.unlink(path)
        except OSError:
            pass
        with _META_CACHE_LOCK:
            _META_CACHE.pop(path, None)


def save_layer(dirpath, data, token, epochs, max_layers=None):
    """Append one layer to the bundle at `dirpath`. Returns a status string:

      "ok"    written as the next layer
      "dup"   every epoch is already stored in this generation — nothing to add
      "stale" `token` is OLDER than what is stored — refused (never regress)
      "full"  the bundle already has `max_layers` layers
      "error" the write failed

    A NEWER token first deletes all older-generation layers (data from another
    forecast cycle must not be mixed in). Run in the cache-writer process."""
    max_layers = MAX_LAYERS if max_layers is None else max_layers
    token = None if token is None else float(token)
    with _SAVE_LOCK:
        existing = layers(dirpath)
        toks = [t for t in (token_of(m) for _, _, m in existing) if t is not None]
        newest = max(toks) if toks else None
        if token is not None and newest is not None:
            if token < newest - TOKEN_EPS:
                return "stale"
            if token > newest + TOKEN_EPS:
                _delete_layers(existing)
                existing = []
        if token is None and newest is not None:
            return "stale"       # an unversioned layer can't join a versioned one
        have = set()
        for _s, _p, m in existing:
            have.update(int(e) for e in m["epochs"])
        if not set(int(e) for e in epochs) - have:
            return "dup"
        if len(existing) >= max_layers:
            return "full"
        try:
            os.makedirs(dirpath, exist_ok=True)
            tmp = os.path.join(dirpath, f".tmp.{os.getpid()}.{threading.get_ident()}")
            with open(tmp, "wb") as f:
                f.write(data)
            try:
                seq = max((s for s, _ in layer_paths(dirpath)), default=0)
                while True:
                    seq += 1
                    final = os.path.join(dirpath, f"{seq:05d}.bin.gz")
                    try:
                        os.link(tmp, final)       # exclusive claim; atomic publish
                        break
                    except FileExistsError:
                        continue
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return "ok"
        except OSError:
            logger.warning("bundle layer write failed in %s", dirpath, exc_info=True)
            return "error"
