import logging
import os
import sys

from fastapi import FastAPI
from uvicorn import run

from . import api, app

logger = logging.getLogger()

def server():

    logroot = os.environ.get("LOG_DIR", "/tmp")
    filename = os.path.join(logroot, "weather_radar.log")
    logging.basicConfig(
        filename=filename,
        level=getattr(logging, str(os.environ.get("LOG_LEVEL")), logging.WARNING)
    )

    logger.addHandler(logging.StreamHandler(sys.stdout))

    server = FastAPI()

    server.include_router(api.router)
    server.include_router(app.router)

    return server

def main():
    run(server(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
