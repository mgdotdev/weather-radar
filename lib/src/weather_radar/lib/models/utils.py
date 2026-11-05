import re

CAMEL_TO_KEBAB = re.compile(r'(?<!^)(?=[A-Z])')
KEBAB_TO_CAMEL = re.compile(r'(?<!\A)-(?=[a-zA-Z])',re.X)


def to_camel_case(value):
    tokens = KEBAB_TO_CAMEL.split(value)
    first = tokens.pop(0)
    # Lowercase only the first character so already-camelCase input (with no
    # hyphens to split on) is returned unchanged, e.g. "snowfallAmount" ->
    # "snowfallAmount" rather than "snowfallamount".
    response = first[:1].lower() + first[1:]
    for remain in tokens:
        # Uppercase only the leading character; preserve the rest so existing
        # casing in a token isn't flattened.
        response += remain[:1].upper() + remain[1:]
    return response


def to_kebab_case(value):
    return CAMEL_TO_KEBAB.sub('_', value).lower()


# Attributes that are physically non-negative (accumulations, rates, counts,
# probabilities, speeds). A CubicSpline fit through these can undershoot below
# zero between knots (Runge-type oscillation), producing spurious negative
# values — e.g. "negative snowfall". Predictions for these are clamped to >= 0
# so the bogus negatives never reach the color scale, legend, readout, or the
# heatmap's transparent-at-zero floor. Keyed by the kebab-case model attribute.
NON_NEGATIVE_ATTRIBUTES = frozenset({
    "quantitative-precipitation",
    "snowfall-amount",
    "ice-accumulation",
    "probability-of-precipitation",
    "relative-humidity",
    "sky-cover",
    "wind-speed",
    "wind-gust",
    "visibility",
})


def clamp_value(attribute, value):
    """Clamp a predicted value to its physical range for `attribute` (kebab
    case). Currently only enforces non-negativity; returns `value` unchanged
    for attributes with no constraint (e.g. temperature, which is legitimately
    negative). `value` may be None (passed through)."""
    if value is None:
        return value
    if attribute in NON_NEGATIVE_ATTRIBUTES and value < 0:
        return 0.0
    return value


class BoundsError(Exception): ...


def source_mtime(path):
    """Return the content-change timestamp for NOAA `path` (e.g.
    "/gridpoints/WFO/x,y"), or None if it isn't cached.

    This is the change-signal for fitted derived artifacts (splines). It reads
    the cache-writer's change-MARKER sidecar (``data.json.changed``), whose
    mtime the writer bumps ONLY when a refetch returns genuinely different
    content — unchanged refreshes (which still extend freshness) leave it
    alone. So a spline is stale IFF this marker's mtime has advanced past the
    value observed at fit time. Falls back to the data.json mtime for legacy
    caches with no marker yet. A single os.stat — cheap enough for a
    per-request freshness gate (no JSON parse of the large doc).

    Imported lazily so this module (pure helpers) has no hard dependency on
    the cache layer.
    """
    import os
    from ..connection import PathObj
    data_file = PathObj(path).tempfile
    marker = data_file + ".changed"
    try:
        return os.path.getmtime(marker)
    except OSError:
        pass
    # Legacy (pre-marker) cache: fall back to the data file's mtime. Less
    # precise (bumps on every refresh) but never misses a real change.
    try:
        return os.path.getmtime(data_file)
    except OSError:
        return None
