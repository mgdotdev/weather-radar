from fastapi import APIRouter

from . import models
from . import admin
from . import geocode

router = APIRouter(prefix="/api")
router.include_router(models.router)
router.include_router(admin.router)
router.include_router(geocode.router)


@router.get("/")
def root():
    return {
        "response": "This is the root of the weather-radar API"
    }
