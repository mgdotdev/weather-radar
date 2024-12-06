import os
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

@router.get("/map/models/{model}/", response_class=HTMLResponse)
def map(model: str, width: float, height: float, lat: float, lon: float, dt: int):
    with open(MAP_HTML) as f:
        template = Template(f.read())
    html = template.render({
        "starting_longitude": lon,
        "starting_latitude": lat,
        "weather_model": f"\"{model}\"",
        "width": width,
        "height": height,
        "dt": dt,
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

