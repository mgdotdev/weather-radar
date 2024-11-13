import inspect
from fastapi import APIRouter

from . import precipitation

router = APIRouter(prefix="/models")
router.include_router(precipitation.router)


@router.get("/")
def models():
    routes = (
        route for route in router.routes if route.name != models.__name__
    )
    return {
        route.path: {
            "query": str(inspect.signature(route.endpoint))
        } for route in routes
    }


