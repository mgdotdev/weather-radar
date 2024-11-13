import os
import concurrent.futures
import multiprocessing
import math
from functools import cached_property

from .connection import NOAAConnection, PathObj, get_and_write

lock = multiprocessing.Lock()


class Coordinate:
    def __init__(self, x, y):
        self.x = round(x, 7)
        self.y = round(y, 7)

    def __str__(self):
        return f'{self.y},{self.x}'

    def __repr__(self):
        return f'"{str(self)}"'


class MapCoordinate(Coordinate):
    def __init__(self, lat, lon):
        super().__init__(x=lon, y=lat)

    @property
    def lat(self):
        return self.y

    @property
    def lon(self):
        return self.x


class NOAAGridpoint:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f'{self.id}/{self.x},{self.y}'

    def __repr__(self) -> str:
        return f'"{str(self)}"'


class CoordinateArea:

    def __init__(self, center: MapCoordinate, width: float, height: float):
        self.center = center
        self.width = max(width, 1.25)
        self.height = max(height, 1.25)

    @cached_property
    def center_gridpoint(self):
        return gridpoint_from_map_coordinate(self.center)

    @cached_property
    def center_id(self):
        return self.center_gridpoint.id

    @property
    def map_coordinates(self):

        dx = dkm_to_dlat(self.width)
        dy = dkm_to_dlon(self.height, self.center.lat)

        x = self.center.lon
        y = self.center.lat

        x -= dx / 2
        y -= dy / 2

        return (
            MapCoordinate(lat=i, lon=j)
            for i in arange(y, y+dy, dkm_to_dlon(1.25, self.center.lat))
            for j in arange(x, x+dx, dkm_to_dlat(1.25))
        )


    @cached_property
    def gridpoints(self):
        points = self.map_coordinates

        points = set(
            gridpoint_from_map_coordinate(p)
            for p in points
        )

        points = sorted(points, key = str)

        return points


def cache_coordinate_area(area: CoordinateArea):
    paths = (f"/points/{c}" for c in area.map_coordinates)
    path_objs = (PathObj(p) for p in paths)
    with lock:
        conn = NOAAConnection()
        missings = [p for p in path_objs if not os.path.isfile(p.tempfile)]
        with concurrent.futures.ThreadPoolExecutor() as exec:
            jobs = [
                exec.submit(get_and_write, m, conn.session)
                for m in missings
            ]
            concurrent.futures.wait(jobs)


    paths = (f"/gridpoints/{c}" for c in area.gridpoints)
    path_objs = (PathObj(p) for p in paths)
    with lock:
        conn = NOAAConnection()
        missings = [p for p in path_objs if not os.path.isfile(p.tempfile)]
        with concurrent.futures.ThreadPoolExecutor() as exec:
            jobs = [
                exec.submit(get_and_write, m, conn.session)
                for m in missings
            ]
            concurrent.futures.wait(jobs)


def gridpoint_from_map_coordinate(coordinate: MapCoordinate):
    conn = NOAAConnection()
    resp = conn.get(f"/points/{coordinate}")
    resp = resp["properties"]
    return NOAAGridpoint(resp["gridId"], resp["gridX"], resp["gridY"])


def dkm_to_dlat(km):
    return km / 110.574


def dkm_to_dlon(km, lat):
    return km / (111.320 * math.cos(math.radians(lat)))


def arange(start, stop, step):
    r = []
    while start < stop:
        r.append(start)
        start += step
    return r


