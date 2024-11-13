from datetime import datetime
from weather_radar.lib.area import MapCoordinate, CoordinateArea
from weather_radar.lib.models.precipitation import PrecipitationEnsemble, PrecipitationModel


class TestPrecipitationModel:
    def test_precipitation_ensemble(self):
        coord = MapCoordinate(47.6763, -116.7798)
        area = CoordinateArea(coord, 40, 10)
        grid = PrecipitationEnsemble(area).predict(datetime.now(), verbose=True)
        assert grid

    def test_precipitation_model(self):
        coord = MapCoordinate(47.6763, -116.7798)
        model = PrecipitationModel.from_map_coordinate(coord)
        resp = model.predict(datetime.now(), verbose=True)
        assert resp

