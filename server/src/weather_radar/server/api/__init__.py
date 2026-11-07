from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


class _WatchBody(BaseModel):
    url: str


@router.post("/hotcache")
def add_to_hotcache(body: _WatchBody):
    """Add a URL to the hot cache. Gated by the ACCESS TOKEN (this lives under
    /api/, which the access-control middleware protects) — NOT the admin
    session: any authenticated app user can warm a view they can see. Listing
    and REVOKING the watch-list remain admin-only under /api/admin/hotcache."""
    from weather_radar.lib import hotcache
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        result = hotcache.watch(url)
    except hotcache.SchedulerUnavailable:
        raise HTTPException(status_code=503, detail="hot-cache scheduler unavailable")
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "could not watch url"))
    return result
