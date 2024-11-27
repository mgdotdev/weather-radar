from datetime import datetime
from weather_radar.lib.area import MapCoordinate, CoordinateArea
from weather_radar.lib.models.accumulation import AccumulationEnsemble, AccumulationModel


class TestAccumulationModel:
    def test_precipitation_ensemble(self):
        coord = MapCoordinate(47.6763, -116.7798)
        area = CoordinateArea(coord, 40, 10)
        grid = AccumulationEnsemble(area, "quantitative-precipitation").predict(datetime.now(), verbose=True)
        assert grid

    def test_precipitation_model(self):
        coord = MapCoordinate(47.6763, -116.7798)
        model = AccumulationModel.from_map_coordinate(coord, "quantitative-precipitation")
        resp = model.predict(datetime.now(), verbose=True)
        assert resp

