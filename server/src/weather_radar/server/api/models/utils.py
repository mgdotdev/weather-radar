from datetime import datetime, timedelta
from typing import Iterable

time_t = float|str|None

GUARD = 10_000

def _time_generator(dtime, dt_td, counts: int = -1):
    count = 0
    while count != counts and count < GUARD:
        yield dtime + (dt_td * count)
        count += 1


def datetime_from_time(time: time_t, dt=0, counts: int = -1) -> datetime|Iterable[datetime]:
    if time is None:
        dtime = datetime.now()
    else:
        try:
            dtime = float(time)
        except ValueError:
            dtime = datetime.fromisoformat(str(time))
        else:
            dtime = datetime.fromtimestamp(dtime)
    if not dt:
        return dtime
    dt_td = timedelta(seconds=float(dt))
    return _time_generator(dtime, dt_td, counts)

