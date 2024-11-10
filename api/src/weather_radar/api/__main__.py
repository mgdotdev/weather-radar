from fastapi import FastAPI
from uvicorn import run

from . import v1


def main():

    app = FastAPI()
    app.include_router(v1.router)

    run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
