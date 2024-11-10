from functools import cached_property

from .connection import NOAAConnection


class Coordinate:
    def __init__(self, x, y):
        self.x = round(x, 4)
        self.y = round(y, 4)

    def __str__(self):
        return f'{self.x},{self.y}'

    def __repr__(self):
        return f'"{str(self)}"'


class MapCoordinate(Coordinate):
    def __init__(self, lat, lon):
        super().__init__(x=lat, y=lon)

    @property
    def lat(self):
        return self.y

    @property
    def lon(self):
        return self.x



class CoordinateArea:
    """each gridpoint is appx. 2.5 km square, so unit conversion would be the
    width (km) * (1 unit / 2.5km) = width (units)"""
    conversion_rate = 2.5

    def __init__(self, center: MapCoordinate, width: float, height: float):
        self.center = center
        self.width = width
        self.height = height

    @cached_property
    def center_gridpoint(self):
        return center_gridpoint_from_map_coordinate(self.center)

    @cached_property
    def center_id(self):
        return center_id_from_map_coordinate(self.center)

    @cached_property
    def gridpoints(self):
        x, y = self.center_gridpoint
        dx = self.width / self.conversion_rate
        dy = self.height / self.conversion_rate

        # make sure there's at least one
        dx = max(dx, 2)
        dy = max(dy, 2)

        x -= dx // 2
        y -= dy // 2
        return [
            Coordinate(x=j, y=i)
            for i in range(int(y), int(y + dy))
            for j in range(int(x), int(x + dx))
        ]


def center_id_from_map_coordinate(coordinate: MapCoordinate):
    conn = NOAAConnection()
    resp = conn.get(f"/points/{coordinate}")
    return resp["properties"]["gridId"]


def center_gridpoint_from_map_coordinate(coordinate: MapCoordinate):
    conn = NOAAConnection()
    resp = conn.get(f"/points/{coordinate}")
    resp = resp["properties"]
    return resp["gridX"], resp["gridY"]
