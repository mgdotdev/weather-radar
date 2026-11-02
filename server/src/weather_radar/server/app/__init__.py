import os
import json
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from jinja2 import Template



HERE = os.path.abspath(os.path.dirname(__file__))
MAP_HTML = os.path.join(HERE, "map.html")
GRAPH_HTML = os.path.join(HERE, "graph.html")

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
#   MODEL_CATALOG  - every supported layer: kebab-case name -> (kind, label).
#                    `kind` selects the ensemble ("observation" for
#                    instantaneous values, "accumulation" for cumulative storm
#                    totals). Non-numeric NOAA layers ("weather", "hazards")
#                    are intentionally excluded since they can't be contoured.
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
    0.0, "rgb(247, 251, 255)",
    0.25, "rgb(198, 219, 239)",
    0.5, "rgb(107, 174, 214)",
    0.75, "rgb(33, 113, 181)",
    1.0, "rgb(8, 48, 107)",
]
_RAMP_TEAL = [
    0.0, "rgb(247, 252, 253)",
    0.25, "rgb(204, 236, 230)",
    0.5, "rgb(102, 194, 164)",
    0.75, "rgb(44, 127, 184)",
    1.0, "rgb(8, 64, 129)",
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
MODEL_CATALOG = {
    # Temperature family
    "temperature": (_OBSERVATION, "Temperature"),
    "dewpoint": (_OBSERVATION, "Dewpoint"),
    "apparent-temperature": (_OBSERVATION, "Apparent Temperature"),
    "max-temperature": (_OBSERVATION, "High Temperature"),
    "min-temperature": (_OBSERVATION, "Low Temperature"),
    "heat-index": (_OBSERVATION, "Heat Index"),
    "wind-chill": (_OBSERVATION, "Wind Chill"),
    # Moisture / precipitation
    "relative-humidity": (_OBSERVATION, "Relative Humidity"),
    "probability-of-precipitation": (_OBSERVATION, "Probability of Precipitation"),
    "quantitative-precipitation": (_ACCUMULATION, "Precipitation"),
    "snowfall-amount": (_ACCUMULATION, "Snowfall"),
    "ice-accumulation": (_ACCUMULATION, "Ice Accumulation"),
    # Wind / sky
    "wind-speed": (_OBSERVATION, "Wind Speed"),
    "wind-gust": (_OBSERVATION, "Wind Gust"),
    "sky-cover": (_OBSERVATION, "Sky Cover"),
    # Other continuous fields
    "pressure": (_OBSERVATION, "Pressure"),
    "visibility": (_OBSERVATION, "Visibility"),
}

# Optional bespoke styling per model (seed step + ramp). Anything omitted uses
# the neutral ramp and a generic seed step (auto-scaling handles the rest).
_CUSTOM_STYLE = {
    "temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP},
    "apparent-temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP},
    "max-temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP},
    "min-temperature": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP},
    "heat-index": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP},
    "wind-chill": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP},
    "dewpoint": {"contour_step": 1.0, "color_ramp": _RAMP_DIVERGING_TEMP},
    "snowfall-amount": {"contour_step": 2.0, "color_ramp": _RAMP_BLUES},
    "quantitative-precipitation": {"contour_step": 0.5, "color_ramp": _RAMP_TEAL},
    "ice-accumulation": {"contour_step": 0.25, "color_ramp": _RAMP_BLUES},
}

_DEFAULT_STYLE = {"contour_step": 1.0, "color_ramp": _RAMP_NEUTRAL}


def _profile(model):
    """Assemble a full profile (kind, label, contour_step, color_ramp) for a
    model name, or None if it isn't a known contourable layer."""
    entry = MODEL_CATALOG.get(model)
    if entry is None:
        return None
    kind, label = entry
    style = _CUSTOM_STYLE.get(model, _DEFAULT_STYLE)
    return {"kind": kind, "label": label, **style}


# Fallback profile for an unknown attribute: render as an observation with the
# neutral ramp so the map still shows something sensible.
DEFAULT_PROFILE = {"kind": _OBSERVATION, "label": "Value", **_DEFAULT_STYLE}



@router.get("/map/models/{model}/", response_class=HTMLResponse)
def map(model: str, width: float, height: float, lat: float, lon: float, dt: int, kind: str = None, step: float = None, speed: float = 1.0):
    # Model names are kebab-case end-to-end (URL, GeoJSON property key, API
    # params). The data layer converts to the camelCase NOAA attribute only
    # when reading the gridpoint (see models.utils.to_camel_case), so nothing
    # here needs to convert.
    profile = _profile(model) or DEFAULT_PROFILE
    # The data kind (which ensemble fetches the attribute) is derived from the
    # model's profile. An explicit `kind` query param may still override it.
    kind = kind or profile["kind"]
    # Contour interval: the value delta between adjacent contour lines. When
    # the user hasn't specified `step`, the client auto-scales it to the data's
    # global range (auto_step=True); the profile value is only a seed/fallback.
    auto_step = step is None
    step = profile["contour_step"] if step is None else step
    step = max(0.1, min(float(step), 100.0))
    # Playback speed multiplier: higher = faster animation (shorter interval
    # between frames). Clamped to a sensible range.
    speed = max(0.1, min(float(speed), 10.0))
    # Options for the in-legend model selector: every supported NOAA layer and
    # its friendly label. (Kind is derived per-model, so it isn't separately
    # user-selectable.)
    model_options = [
        {"value": key, "label": label}
        for key, (_kind, label) in MODEL_CATALOG.items()
    ]
    with open(MAP_HTML) as f:
        template = Template(f.read())
    html = template.render({
        "starting_longitude": lon,
        "starting_latitude": lat,
        "weather_model": f"\"{model}\"",
        "width": width,
        "height": height,
        "dt": dt,
        "kind": kind,
        "contour_step": step,
        "auto_step": "true" if auto_step else "false",
        "playback_speed": speed,
        "contour_color_ramp": json.dumps(profile["color_ramp"]),
        "model_options": json.dumps(model_options),
        "current_model": model,
    })
    return html


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

