import os
import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from jinja2 import Template



HERE = os.path.abspath(os.path.dirname(__file__))
MAP_HTML = os.path.join(HERE, "map.html")
GRAPH_HTML = os.path.join(HERE, "graph.html")
FORECAST_HTML = os.path.join(HERE, "forecast.html")
ADMIN_HTML = os.path.join(HERE, "admin.html")

router = APIRouter()

DARK_MODE = """
  <style type="text/css">
  .plot-container {
    filter: invert(75%) hue-rotate(180deg);
  }
  </style>
"""


# ---------------------------------------------------------------------------
# Model registry
#
# The map can render any numeric NOAA gridpoint layer as a contour field. Two
# pieces drive this:
#   MODEL_CATALOG  - every supported layer: dropdown key -> (kind, label,
#                    attribute). `kind` selects the ensemble ("observation" for
#                    instantaneous/per-interval values, "accumulation" for
#                    cumulative storm totals). `attribute` is the NOAA gridpoint
#                    attribute to read, usually equal to the key but distinct
#                    where one attribute is offered under several keys (the
#                    precip/snow/ice totals also appear as per-interval rates).
#                    Non-numeric NOAA layers ("weather", "hazards") are
#                    intentionally excluded since they can't be contoured.
#   _CUSTOM_STYLE  - optional per-model styling (color_ramp, seed contour_step)
#                    for the headline models; anything not listed falls back to
#                    a neutral sequential ramp.
#
# Units are NOT specified here: they flow through from the data (NOAA's `uom`,
# e.g. degF after the temperature wrapper, inches for accumulation, or the
# native unit otherwise), so the legend/readout always reflect real units.
#
# The contour step auto-scales to each model's value range client-side, so the
# per-model `contour_step` below is only a seed/fallback.
# ---------------------------------------------------------------------------

# Reusable sequential color ramps (opaque; layer applies uniform transparency).
_RAMP_DIVERGING_TEMP = [
    0.0,   "rgb(69, 117, 180)",
    0.125, "rgb(96, 141, 186)",
    0.25,  "rgb(123, 165, 180)",
    0.375, "rgb(150, 183, 166)",
    0.5,   "rgb(180, 196, 150)",
    0.625, "rgb(205, 192, 130)",
    0.75,  "rgb(222, 173, 110)",
    0.875, "rgb(227, 146, 95)",
    1.0,   "rgb(222, 119, 86)",
]
_RAMP_BLUES = [
    0.0, "rgb(158, 202, 225)",
    0.25, "rgb(107, 174, 214)",
    0.5, "rgb(66, 146, 198)",
    0.75, "rgb(33, 113, 181)",
    1.0, "rgb(8, 48, 107)",
]
_RAMP_TEAL = [
    0.0, "rgb(153, 216, 201)",
    0.25, "rgb(102, 194, 164)",
    0.5, "rgb(65, 174, 118)",
    0.75, "rgb(44, 127, 184)",
    1.0, "rgb(8, 64, 129)",
]
# Cloud-shading ramp for sky cover: a near-white haze at low cover deepening
# to a flat overcast grey. Rendered as a heatmap wash (opacity scales with
# cover) so it reads like cloud shadow over the basemap rather than banded
# contour zones.
_RAMP_CLOUD = [
    0.0, "rgb(245, 245, 245)",
    0.5, "rgb(180, 182, 186)",
    1.0, "rgb(110, 112, 118)",
]

# Neutral sequential ramp (light grey -> deep purple) used for any layer
# without a bespoke ramp, so every model still renders legibly.
_RAMP_NEUTRAL = [
    0.0, "rgb(247, 247, 247)",
    0.25, "rgb(204, 204, 220)",
    0.5, "rgb(158, 154, 200)",
    0.75, "rgb(106, 81, 163)",
    1.0, "rgb(63, 0, 125)",
]

# EPA AQI category colors as a SMOOTH gradient over the 0..300 displayed AQI
# range (the /aqi endpoints fix min=0, max=300). Blends continuously through
# the category colors at their threshold positions (t/300) — a heatmap wash,
# not hard category bands.
#   Good            0   green  (0,228,0)
#   Moderate       50   yellow (255,255,0)
#   USG           100   orange (255,126,0)
#   Unhealthy     150   red    (255,0,0)
#   Very Unhealthy 200  purple (143,63,151)
#   (300)               maroon (126,0,35)
_RAMP_AQI = [
    0.0,            "rgb(0, 228, 0)",
    50.0 / 300.0,   "rgb(255, 255, 0)",
    100.0 / 300.0,  "rgb(255, 126, 0)",
    150.0 / 300.0,  "rgb(255, 0, 0)",
    200.0 / 300.0,  "rgb(143, 63, 151)",
    1.0,            "rgb(126, 0, 35)",
]

# (kind, label) for every contourable NOAA gridpoint layer we support, keyed
# by the kebab-case name used in URLs. Cumulative layers use "accumulation".
#
# This is a CURATED subset of NOAA's gridpoint layers, chosen because they are
# continuous scalar fields that contour well and are broadly available.
# Intentionally excluded:
#   * direction fields (wind/wave/swell direction) — 0..360° wraps around, so
#     contouring produces garbage bands across the 359->0 seam;
#   * wave/swell layers — coastal-only, absent inland (empty map);
#   * dimensionless / categorical indices and the potentialOf* family — niche,
#     often sparse, and poor as continuous contours.
_ACCUMULATION = "accumulation"
_OBSERVATION = "observation"
_AQI = "aqi"  # NAQFC gridded air-quality forecast; a separate (non-NOAA) data
              # path sampled directly from GRIB2 (see lib/aqi.py + /aqi endpoints).

# Each entry is (kind, label, attribute):
#   kind      - which ensemble fetches the data ("observation" for
#               instantaneous/per-interval values, "accumulation" for a
#               cumulative running total).
#   label     - friendly name shown in the in-legend model selector.
#   attribute - the kebab-case NOAA gridpoint attribute to read. This is
#               USUALLY the same as the catalog key, but is kept separate so a
#               single NOAA attribute can be exposed under multiple dropdown
#               keys with different kinds — e.g. the precipitation/snow/ice
#               amounts are offered BOTH as a cumulative "accumulation" total
#               and as a per-interval "observation" rate, both reading the same
#               underlying NOAA attribute.
MODEL_CATALOG = {
    # Temperature family
    "temperature": (_OBSERVATION, "Temperature", "temperature"),
    "dewpoint": (_OBSERVATION, "Dewpoint", "dewpoint"),
    "apparent-temperature": (_OBSERVATION, "Apparent Temperature", "apparent-temperature"),
    "max-temperature": (_OBSERVATION, "High Temperature", "max-temperature"),
    "min-temperature": (_OBSERVATION, "Low Temperature", "min-temperature"),
    "heat-index": (_OBSERVATION, "Heat Index", "heat-index"),
    "wind-chill": (_OBSERVATION, "Wind Chill", "wind-chill"),
    # Moisture / precipitation (cumulative totals)
    "relative-humidity": (_OBSERVATION, "Relative Humidity", "relative-humidity"),
    "probability-of-precipitation": (_OBSERVATION, "Probability of Precipitation", "probability-of-precipitation"),
    "quantitative-precipitation": (_ACCUMULATION, "Precipitation (total)", "quantitative-precipitation"),
    "snowfall-amount": (_ACCUMULATION, "Snowfall (total)", "snowfall-amount"),
    "ice-accumulation": (_ACCUMULATION, "Ice Accumulation (total)", "ice-accumulation"),
    # Moisture / precipitation (per-interval rate) — same NOAA attributes as
    # the totals above, but rendered as the raw per-interval amount instead of
    # a running sum.
    "precipitation-rate": (_OBSERVATION, "Precipitation (per interval)", "quantitative-precipitation"),
    "snowfall-rate": (_OBSERVATION, "Snowfall (per interval)", "snowfall-amount"),
    "ice-accumulation-rate": (_OBSERVATION, "Ice Accumulation (per interval)", "ice-accumulation"),
    # Wind / sky
    "wind-speed": (_OBSERVATION, "Wind Speed", "wind-speed"),
    "wind-gust": (_OBSERVATION, "Wind Gust", "wind-gust"),
    "sky-cover": (_OBSERVATION, "Sky Cover", "sky-cover"),
    # Other continuous fields
    "pressure": (_OBSERVATION, "Pressure", "pressure"),
    "visibility": (_OBSERVATION, "Visibility", "visibility"),
    # Air quality (NAQFC gridded hourly forecast). `attribute` is the pollutant
    # key passed to the /aqi endpoints (pm25 | ozone). kind=_AQI routes the
    # client to the AQI data path instead of the NOAA ensemble path.
    "air-quality-pm25": (_AQI, "Air Quality (PM2.5)", "pm25"),
    "air-quality-ozone": (_AQI, "Air Quality (Ozone)", "ozone"),
}

# Optional bespoke styling per model (seed step + ramp). Anything omitted uses
# the neutral ramp and a generic seed step (auto-scaling handles the rest).
#
# NOTE: isoband (contour) rendering was removed — every model now renders as a
# heatmap wash. The `render` key / _RENDER_* constants are retained only so an
# explicit `?render=` override and the client's render dispatch keep working;
# the profile default is always heatmap (see _profile).
_RENDER_ISOBAND = "isoband"
_RENDER_HEATMAP = "heatmap"
# Per-model heatmap opacity multiplier (applied on top of the value-driven
# alpha, client-side). DENSE/continuous fields (temperature, humidity, sky,
# AQI) cover essentially the WHOLE area, so at full alpha they completely
# occlude the basemap — they read better as a lighter wash you can see the map
# through. SPARSE fields (snow/precip/ice amounts + rates) are mostly empty with
# isolated features, so they stay fully opaque where present. Omitted => 1.0.
_OPACITY_DENSE = 0.6
_CUSTOM_STYLE = {
    "temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP, "opacity": _OPACITY_DENSE},
    "apparent-temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP, "opacity": _OPACITY_DENSE},
    "max-temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP, "opacity": _OPACITY_DENSE},
    "min-temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP, "opacity": _OPACITY_DENSE},
    "heat-index": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP, "opacity": _OPACITY_DENSE},
    "wind-chill": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP, "opacity": _OPACITY_DENSE},
    "dewpoint": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP, "opacity": _OPACITY_DENSE},
    "snowfall-amount": {"contour_step": 2.0, "color_ramp": _RAMP_BLUES},
    "quantitative-precipitation": {"contour_step": 0.5, "color_ramp": _RAMP_TEAL},
    "ice-accumulation": {"contour_step": 0.25, "color_ramp": _RAMP_BLUES},
    # Per-interval rate variants reuse their total counterpart's ramp but
    # render as an intensity heatmap (radar-style wash) instead of isobands. A
    # smaller seed step suits the per-interval amounts (a single interval is
    # only a fraction of the storm total); auto-scaling refines it anyway.
    "precipitation-rate": {"contour_step": 0.1, "color_ramp": _RAMP_TEAL, "render": _RENDER_HEATMAP},
    "snowfall-rate": {"contour_step": 0.5, "color_ramp": _RAMP_BLUES, "render": _RENDER_HEATMAP},
    "ice-accumulation-rate": {"contour_step": 0.1, "color_ramp": _RAMP_BLUES, "render": _RENDER_HEATMAP},
    # Sky cover (0-100% cloud): a grey heatmap wash whose opacity scales with
    # cover reads like cloud shadow over the basemap, rather than contour bands
    # that imply hard cloud edges. Seed step of 10% ~= one okta; auto-scaling
    # refines it.
    "sky-cover": {"contour_step": 10.0, "color_ramp": _RAMP_CLOUD, "render": _RENDER_HEATMAP, "opacity": _OPACITY_DENSE},
    # Air quality: the stepwise EPA AQI category ramp as a heatmap wash (smooth
    # field, category-colored). Rendered as heatmap because the IDW field reads
    # like Apple-style AQI shading; the stepwise ramp keeps the category colors
    # crisp. contour_step is unused for heatmaps (kept as a seed). Dense field
    # (covers the whole area), so dimmed so the basemap shows through.
    "air-quality-pm25": {"contour_step": 25.0, "color_ramp": _RAMP_AQI, "render": _RENDER_HEATMAP, "opacity": _OPACITY_DENSE},
    "air-quality-ozone": {"contour_step": 25.0, "color_ramp": _RAMP_AQI, "render": _RENDER_HEATMAP, "opacity": _OPACITY_DENSE},
}

_DEFAULT_STYLE = {"contour_step": 1.0, "color_ramp": _RAMP_NEUTRAL}


def _profile(model):
    """Assemble a full profile (kind, label, attribute, contour_step,
    color_ramp) for a model name, or None if it isn't a known contourable
    layer. `attribute` is the kebab-case NOAA attribute to read, which may
    differ from the dropdown key (see MODEL_CATALOG)."""
    entry = MODEL_CATALOG.get(model)
    if entry is None:
        return None
    kind, label, attribute = entry
    style = _CUSTOM_STYLE.get(model, _DEFAULT_STYLE)
    # All models render as a heatmap wash (isobands were dropped). The `render`
    # query param can still override per-request, but the profile default — and
    # any per-model style `render` — is heatmap.
    return {
        "kind": kind,
        "label": label,
        "attribute": attribute,
        "render": _RENDER_HEATMAP,
        **style,
    }


# Fallback profile for an unknown attribute: render as an observation with the
# neutral ramp so the map still shows something sensible. `attribute` is filled
# in by the caller (it defaults to the requested model name).
DEFAULT_PROFILE = {"kind": _OBSERVATION, "label": "Value", "render": _RENDER_HEATMAP, **_DEFAULT_STYLE}



@router.get("/map/models/{model}/", response_class=HTMLResponse)
def map(model: str, width: float, height: float, lat: float, lon: float, dt: int, kind: str = None, step: float = None, speed: float = 1.0, render: str = None):
    # Model names are kebab-case end-to-end (URL, GeoJSON property key, API
    # params). The data layer converts to the camelCase NOAA attribute only
    # when reading the gridpoint (see models.utils.to_camel_case), so nothing
    # here needs to convert.
    profile = _profile(model) or {**DEFAULT_PROFILE, "attribute": model}
    # The data kind (which ensemble fetches the attribute) is derived from the
    # model's profile. An explicit `kind` query param may still override it.
    kind = kind or profile["kind"]
    # The NOAA attribute to actually read. This is usually the same as the
    # dropdown/URL `model` key, but differs for the per-interval rate variants
    # (e.g. model "precipitation-rate" reads attribute
    # "quantitative-precipitation" as an observation). All API calls the client
    # makes use this attribute (it's injected as the JS `model` var below), so
    # the GeoJSON property key and the client's readout stay in sync; `model`
    # (the URL key) is preserved separately for the dropdown selection.
    attribute = profile["attribute"]
    # Overlay type: isobands by default, heatmap for the rate layers. An
    # explicit `render` query param may override the profile default.
    render_type = render or profile["render"]
    if render_type not in (_RENDER_ISOBAND, _RENDER_HEATMAP):
        render_type = _RENDER_ISOBAND
    # Contour interval: the value delta between adjacent contour lines. When
    # the user hasn't specified `step`, the client auto-scales it to the data's
    # global range (auto_step=True); the profile value is only a seed/fallback.
    auto_step = step is None
    step = profile["contour_step"] if step is None else step
    step = max(0.1, min(float(step), 100.0))
    # Playback speed multiplier: higher = faster animation (shorter interval
    # between frames). Clamped to a sensible range. Max matches the client's
    # speed ladder (SPEED_STEPS tops out at 50x) so a persisted ?speed= value
    # survives a refresh instead of being capped back down.
    speed = max(0.1, min(float(speed), 50.0))
    # Options for the in-legend model selector: every supported NOAA layer and
    # its friendly label. (Kind is derived per-model, so it isn't separately
    # user-selectable.)
    model_options = [
        {"value": key, "label": label}
        for key, (_kind, label, _attr) in MODEL_CATALOG.items()
    ]
    with open(MAP_HTML) as f:
        template = Template(f.read())
    html = template.render({
        "starting_longitude": lon,
        "starting_latitude": lat,
        "weather_model": f"\"{attribute}\"",
        "width": width,
        "height": height,
        "dt": dt,
        "kind": kind,
        "contour_step": step,
        "auto_step": "true" if auto_step else "false",
        "playback_speed": speed,
        "contour_color_ramp": json.dumps(profile["color_ramp"]),
        "heatmap_opacity": profile.get("opacity", 1.0),
        "model_options": json.dumps(model_options),
        "current_model": model,
        "render_type": render_type,
    })
    return html


@router.get("/admin", response_class=HTMLResponse)
def admin():
    """Serve the hot-cache admin page. The page's API calls are themselves
    access-controlled (localhost-only, or ADMIN_TOKEN); this just serves the
    static UI."""
    with open(ADMIN_HTML) as f:
        return f.read()


@router.get("/graph/", response_class=HTMLResponse)
def graph(models: str, lat: float, lon: float, dark_mode: bool = True, dt: int = 450, counts: int = -1, ha: bool = False):
    with open(GRAPH_HTML) as f:
        template = Template(f.read())
    html = template.render({
        "starting_longitude": lon,
        "starting_latitude": lat,
        "weather_models": models,
        "dt": dt,
        "counts": counts,
        "dark_mode": DARK_MODE if dark_mode else "",
        "is_ha": "true" if ha else "false"
    })
    return html


@router.get("/forecast/", response_class=HTMLResponse)
def forecast(model: str, lat: float, lon: float, days: int = 7, name: str = None):
    """A standard N-day (default 7) forecast view for one model at a point.

    The page fetches the model's hourly series (/api/models/{kind}/params/
    {attribute}) and aggregates it into per-day summaries client-side. `model`
    is the dropdown key; its profile supplies the kind/attribute/label/ramp.
    `name` is an optional location label (e.g. a pin name) for the header."""
    profile = _profile(model) or {**DEFAULT_PROFILE, "attribute": model}
    kind = profile["kind"]
    attribute = profile["attribute"]
    with open(FORECAST_HTML) as f:
        template = Template(f.read())
    html = template.render({
        "starting_latitude": lat,
        "starting_longitude": lon,
        "kind": kind,
        "attribute": attribute,
        "model_key": model,
        "label": profile.get("label", model),
        "color_ramp": json.dumps(profile["color_ramp"]),
        "days": max(1, min(int(days), 16)),
        "location_name": json.dumps(name or ""),
    })
    return html

