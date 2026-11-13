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


class TestAreaAxes:
    """width is the LONGITUDE extent, height the LATITUDE extent (the [lon,lat]
    convention the client uses to size a drawn/viewport box). index_bounds must
    map each dimension to its correct axis + conversion factor; a non-square
    box at a mid latitude makes a swap/wrong-factor obviously wrong."""

    def test_nonsquare_box_maps_to_correct_axes(self):
        import math
        from weather_radar.lib.area import LATTICE_STEP_DEG
        lat0 = 47.0
        w_km, h_km = 400.0, 100.0   # width=lon extent, height=lat extent
        area = CoordinateArea(MapCoordinate(lat0, -121.0), w_km, h_km)
        i0, i1, j0, j1 = area.index_bounds
        lat_span_km = (i1 - i0) * LATTICE_STEP_DEG * 110.574
        lon_span_km = (j1 - j0) * LATTICE_STEP_DEG * 111.320 * math.cos(math.radians(lat0))
        # Each measured span should be within ~one lattice cell of the request
        # (floor/ceil snapping adds up to a cell per edge). The longitude span
        # tracks width (~400), the latitude span tracks height (~100).
        cell_km = LATTICE_STEP_DEG * 111.320 * math.cos(math.radians(lat0))
        assert abs(lon_span_km - w_km) <= 2 * cell_km + 1, lon_span_km
        assert abs(lat_span_km - h_km) <= 2 * LATTICE_STEP_DEG * 110.574 + 1, lat_span_km
        # And the aspect ratio is NOT transposed: lon span clearly exceeds lat.
        assert lon_span_km > lat_span_km


class TestRequestedBounds:
    """requested_bounds is the EXACT drawn rectangle (center ± half span, no
    lattice snapping), used to size the user-facing area box for the gridpoints
    source so it matches what the user drew. bounds (lattice-snapped) rounds
    OUTWARD and so is always >= requested_bounds."""

    def test_requested_bounds_matches_drawn_span(self):
        import math
        lat0, lon0 = 48.401829, -116.667964
        w_km, h_km = 34.36, 36.83
        area = CoordinateArea(MapCoordinate(lat0, lon0), w_km, h_km)
        rb = area.requested_bounds
        lon_km_per_deg = 111.320 * math.cos(math.radians(lat0))
        lat_km_per_deg = 110.574
        assert abs((rb["e"] - rb["w"]) * lon_km_per_deg - w_km) < 1e-6
        assert abs((rb["n"] - rb["s"]) * lat_km_per_deg - h_km) < 1e-6
        # Centered on the request.
        assert abs((rb["w"] + rb["e"]) / 2 - lon0) < 1e-9
        assert abs((rb["s"] + rb["n"]) / 2 - lat0) < 1e-9

    def test_requested_bounds_within_lattice_bounds(self):
        # The lattice bounds round outward, so they always contain the exact
        # drawn rectangle.
        area = CoordinateArea(MapCoordinate(48.401829, -116.667964), 34.36, 36.83)
        rb, b = area.requested_bounds, area.bounds
        assert b["w"] <= rb["w"] and b["e"] >= rb["e"]
        assert b["s"] <= rb["s"] and b["n"] >= rb["n"]

