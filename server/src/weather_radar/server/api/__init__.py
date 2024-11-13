from fastapi import APIRouter

from . import models

router = APIRouter(prefix="/api")
router.include_router(models.router)


@router.get("/")
def root():
    return {
        "response": "This is the root of the weather-radar API"
    }
