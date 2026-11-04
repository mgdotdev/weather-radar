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
