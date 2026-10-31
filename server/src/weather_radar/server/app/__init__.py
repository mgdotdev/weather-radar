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


# Per-model rendering profiles for the contour map. Each entry fully describes
# how a NOAA gridpoint attribute is fetched and drawn:
#   kind          -> which ensemble resolves it ("observation" or "accumulation")
#   label         -> human-friendly name (unused server-side; handy for docs)
#   contour_step  -> default value delta between contour lines (live-adjustable)
#   color_ramp    -> sequential [pos, "rgb(...)", ...] ramp for fill + borders
# Units are NOT specified here: they flow through from the model's data
# (NOAA's `uom`, e.g. degF after the temperature wrapper, or native mm for
# accumulation), so the legend/readout always reflect the real units.
#
# Accumulation attributes (snowfall, precipitation) are cumulative "storm
# totals": each frame shows the running sum up to that time, so contours only
# grow over the animation. The fixed global color scale (value_range samples
# the whole lifecycle) keeps the end-of-storm maximum from blowing out early
# frames.
MODEL_PROFILES = {
    "temperature": {
        "kind": "observation",
        "label": "Temperature",
        "contour_step": 1.0,
        # Subtle sequential cool->warm gradient. Opaque colors; transparency is
        # applied uniformly via the fill layer so it never varies per band.
        "color_ramp": [
            0.0,   "rgb(69, 117, 180)",
            0.125, "rgb(96, 141, 186)",
            0.25,  "rgb(123, 165, 180)",
            0.375, "rgb(150, 183, 166)",
            0.5,   "rgb(180, 196, 150)",
            0.625, "rgb(205, 192, 130)",
            0.75,  "rgb(222, 173, 110)",
            0.875, "rgb(227, 146, 95)",
            1.0,   "rgb(222, 119, 86)",
        ],
    },
    "snowfall-amount": {
        "kind": "accumulation",
        "label": "Snowfall",
        # Snow totals (inches) can reach a few feet over a storm; a coarser
        # default step keeps the contour set readable. Adjustable in the legend.
        "contour_step": 2.0,
        # White -> blue sequential ramp.
        "color_ramp": [
            0.0, "rgb(247, 251, 255)",
            0.25, "rgb(198, 219, 239)",
            0.5, "rgb(107, 174, 214)",
            0.75, "rgb(33, 113, 181)",
            1.0, "rgb(8, 48, 107)",
        ],
    },
    "quantitative-precipitation": {
        "kind": "accumulation",
        "label": "Precipitation",
        # Precipitation (inches) totals are typically smaller than snow; a
        # finer default step surfaces light-precip structure.
        "contour_step": 0.5,
        # White -> teal -> blue sequential ramp.
        "color_ramp": [
            0.0, "rgb(247, 252, 253)",
            0.25, "rgb(204, 236, 230)",
            0.5, "rgb(102, 194, 164)",
            0.75, "rgb(44, 127, 184)",
            1.0, "rgb(8, 64, 129)",
        ],
    },
}

# Fallback profile for any unregistered attribute: treat it as an observation
# with a neutral sequential ramp so the map still renders something sensible.
DEFAULT_PROFILE = MODEL_PROFILES["temperature"]


@router.get("/map/models/{model}/", response_class=HTMLResponse)
def map(model: str, width: float, height: float, lat: float, lon: float, dt: int, kind: str = None, step: float = None, speed: float = 1.0):
    # Model names are kebab-case end-to-end (URL, GeoJSON property key, API
    # params). The data layer converts to the camelCase NOAA attribute only
    # when reading the gridpoint (see models.utils.to_camel_case), so nothing
    # here needs to convert.
    profile = MODEL_PROFILES.get(model, DEFAULT_PROFILE)
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

