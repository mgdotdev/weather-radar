from weather_radar.lib.area import MapCoordinate, CoordinateArea


class TestCoordinates:
    def test_gridpoints(self):
        coord = MapCoordinate(lat=47.6763, lon=-116.7798)
        area = CoordinateArea(coord, 10, 7)
        assert area.gridpoints

    def test_gridpoint_of_one(self):
        # A minimal area resolves to a small set of gridpoints. (It is not
        # exactly 1: sample coordinates are snapped to a fixed global lattice
        # so overlapping areas reuse each other's cached /points lookups, which
        # means a minimal area may straddle a few lattice cells / gridpoints.)
        coord = MapCoordinate(lat=47.6763, lon=-116.7798)
        area = CoordinateArea(coord, 1, 1)
        assert 1 <= len(area.gridpoints) <= 4

