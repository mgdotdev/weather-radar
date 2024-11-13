import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from jinja2 import Template


router = APIRouter(prefix="/app")


HERE = os.path.abspath(os.path.dirname(__file__))
HTML = os.path.join(HERE, "app.html")

@router.get("/models/{model}/", response_class=HTMLResponse)
def root(model: str, width: float, height: float, dt: int):
    with open(HTML) as f:
        template = Template(f.read())
    return template.render({
        "starting_longitude": -116.7798,
        "starting_latitude": 47.6763,
        "weather_model": f"\"{model}\"",
        "width": width,
        "height": height,
        "dt": dt,
    })
