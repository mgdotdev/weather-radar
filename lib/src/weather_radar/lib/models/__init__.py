import os

from .observation import ObservationModel, ObservationEnsemble
from .accumulation import AccumulationModel, AccumulationEnsemble

MODEL_MAP = {
    "observation": ObservationModel,
    "observation-ensemble": ObservationEnsemble,
    "accumulation": AccumulationModel,
    "accumulation-ensemble": AccumulationEnsemble
}

def model_from_type(model_type):
    return MODEL_MAP.get(model_type, ObservationModel)


# --- Weather source selection (WR_WEATHER_SOURCE) ---------------------------
# The app can resolve weather layers from either the NDFD GRIB2 files (default)
# or the api.weather.gov /gridpoints JSON API. Both expose the SAME
# ensemble contract (see models.ndfd_model), so switching the source is a
# matter of returning a different ensemble class per `kind`. Selected once by
# the WR_WEATHER_SOURCE env var; see docs/ndfd-roadmap.md.

def weather_source():
    """The configured weather data source: "ndfd" (default), "nbm", or
    "gridpoints"."""
    src = os.environ.get("WR_WEATHER_SOURCE", "ndfd").strip().lower()
    if src == "gridpoints":
        return "gridpoints"
    if src == "nbm":
        return "nbm"
    return "ndfd"


def ensemble_kinds():
    """Map public `kind` -> ensemble class for the CONFIGURED weather source.

    ndfd (default): the GRIB-backed NDFD adapters. nbm: the National Blend of
    Models GRIB adapters (national grid, ~11-day horizon incl. extended-range
    snow/qpf NDFD lacks). gridpoints: the JSON-backed ObservationEnsemble /
    AccumulationEnsemble. All expose the SAME ensemble contract. AQI is NOT here
    — it's a separate data path handled by its own /aqi endpoints regardless of
    weather source."""
    src = weather_source()
    if src == "ndfd":
        from .ndfd_model import NDFDObservationEnsemble, NDFDAccumulationEnsemble
        return {
            "observation": NDFDObservationEnsemble,
            "accumulation": NDFDAccumulationEnsemble,
        }
    if src == "nbm":
        from .nbm_model import NBMObservationEnsemble, NBMAccumulationEnsemble
        return {
            "observation": NBMObservationEnsemble,
            "accumulation": NBMAccumulationEnsemble,
        }
    return {
        "observation": ObservationEnsemble,
        "accumulation": AccumulationEnsemble,
    }
