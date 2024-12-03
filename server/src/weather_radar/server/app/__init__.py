import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from jinja2 import Template


HERE = os.path.abspath(os.path.dirname(__file__))
MAP_HTML = os.path.join(HERE, "map.html")
GRAPH_HTML = os.path.join(HERE, "graph.html")

router = APIRouter()


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
def graph(models: str, lat: float, lon: float, dt: int = 0, counts: int = 0):
    with open(GRAPH_HTML) as f:
        template = Template(f.read())
    html = template.render({
        "starting_longitude": lon,
        "starting_latitude": lat,
        "weather_models": models,
        "dt": dt,
        "counts": counts,
    })
    return html

