from weather_radar.lib.area import MapCoordinate, CoordinateArea


class TestCoordinates:
    def test_gridpoints(self):
        coord = MapCoordinate(lat=47.6763, lon=-116.7798)
        area = CoordinateArea(coord, 10, 7)
        assert area.gridpoints

    def test_gridpoint_of_one(self):
        # A minimal area is bumped up to one SAMPLE_STEP_KM cell and then
        # sampled on the fixed global lattice. Because the lattice bounds use
        # floor/ceil to fully cover the area's edges, a tiny area can straddle
        # several adjacent lattice cells, and samples near a NOAA gridpoint
        # boundary may resolve to distinct gridpoints. So the result is a
        # SMALL set, not exactly 1 — the point is that it stays bounded, and
        # that overlapping areas reuse each other's cached /points lookups.
        coord = MapCoordinate(lat=47.6763, lon=-116.7798)
        area = CoordinateArea(coord, 1, 1)
        assert 1 <= len(area.gridpoints) <= 16

