from fastapi import APIRouter

from . import models
from . import admin
from . import geocode
from . import auth

router = APIRouter(prefix="/api")
router.include_router(models.router)
router.include_router(admin.router)
router.include_router(geocode.router)
router.include_router(auth.router)


@router.get("/")
def root():
    return {
        "response": "This is the root of the weather-radar API"
    }
