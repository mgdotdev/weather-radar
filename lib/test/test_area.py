from weather_radar.lib.area import MapCoordinate, CoordinateArea


class TestCoordinates:
    def test_gridpoints(self):
        coord = MapCoordinate(lat=47.6763, lon=-116.7798)
        area = CoordinateArea(coord, 10, 7)
        assert area.gridpoints
        assert area.center_id

    def test_gridpoint_of_one(self):
        coord = MapCoordinate(lat=47.6763, lon=-116.7798)
        area = CoordinateArea(coord, 1, 1)
        assert len(area.gridpoints) == 4
        assert area.center_id
