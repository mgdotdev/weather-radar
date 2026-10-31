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


# Per-kind heatmap rendering profiles. Each profile supplies the MapLibre
# `heatmap-weight` input->weight stops and the `heatmap-color` density ramp.
# `prop` is the GeoJSON property key the features carry (equal to the model
# name). Precipitation is an accumulation that should fade to transparent at
# zero; temperature is an observation whose cold end is meaningful, so it uses
# a diverging cold->hot ramp with no transparent floor.
HEATMAP_PROFILES = {
    "accumulation": {
        # Precipitation uses a fixed, absolute weight scale: 0 genuinely means
        # "no precipitation", so the range should not adapt to the data.
        "adaptive": False,
        "weight_stops": [0, 0, 0.5, 0.2, 2, 0.5, 5, 0.8, 10, 1],
        "color_ramp": [
            0, "rgba(0, 0, 255, 0)",
            0.2, "rgba(0, 170, 255, 0.6)",
            0.4, "rgba(0, 255, 170, 0.7)",
            0.6, "rgba(170, 255, 0, 0.8)",
            0.8, "rgba(255, 170, 0, 0.9)",
            1, "rgba(255, 0, 0, 1)",
        ],
    },
    "observation": {
        # Temperature (and other observations) have no meaningful absolute
        # zero for colouring, and a local area often spans only a few degrees.
        # `adaptive` tells the client to rescale the weight ramp to each
        # payload's actual min/max so local variation is visible. The
        # `weight_stops` below are only a fallback used before the first
        # payload arrives.
        "adaptive": True,
        "weight_stops": [0, 0, 20, 0.2, 50, 0.4, 70, 0.6, 90, 0.8, 110, 1],
        # A subtle sequential gradient with many closely-spaced stops, so
        # adjacent contour bands differ only slightly. Colors are opaque here;
        # transparency is applied uniformly via the layer's fill-opacity so it
        # never varies between frames/bands.
        "color_ramp": [
            0.0,  "rgb(69, 117, 180)",
            0.125, "rgb(96, 141, 186)",
            0.25, "rgb(123, 165, 180)",
            0.375, "rgb(150, 183, 166)",
            0.5,  "rgb(180, 196, 150)",
            0.625, "rgb(205, 192, 130)",
            0.75, "rgb(222, 173, 110)",
            0.875, "rgb(227, 146, 95)",
            1.0,  "rgb(222, 119, 86)",
        ],
    },
}


@router.get("/map/models/{model}/", response_class=HTMLResponse)
def map(model: str, width: float, height: float, lat: float, lon: float, dt: int, kind: str = "accumulation", step: float = 1.0, speed: float = 1.0):
    profile = HEATMAP_PROFILES.get(kind, HEATMAP_PROFILES["accumulation"])
    # Contour interval: the value delta between adjacent contour lines (e.g.
    # 1.0 degree, 0.5 degree). Clamped to a sane positive range.
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
        "playback_speed": speed,
        "heatmap_adaptive": "true" if profile.get("adaptive") else "false",
        "heatmap_weight_stops": json.dumps(profile["weight_stops"]),
        "heatmap_color_ramp": json.dumps(profile["color_ramp"]),
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

