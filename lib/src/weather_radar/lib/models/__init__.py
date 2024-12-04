from .observation import ObservationModel
from .accumulation import AccumulationModel, AccumulationEnsemble

MODEL_MAP = {
    "observation": ObservationModel,
    "accumulation": AccumulationModel,
    "accumulation-ensemble": AccumulationEnsemble
}

def model_from_type(model_type):
    return MODEL_MAP.get(model_type, ObservationModel)
