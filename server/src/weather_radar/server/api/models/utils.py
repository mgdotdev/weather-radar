from datetime import datetime, timedelta

time_t = float|str|None

def datetime_from_time(time: time_t, dt=0, counts=0) -> datetime|list[datetime]:
    if time is None:
        dtime = datetime.now()
    else:
        try:
            dtime = float(time)
        except ValueError:
            dtime = datetime.fromisoformat(str(time))
        else:
            dtime = datetime.fromtimestamp(dtime)
    if not counts:
        return dtime
    dt_td = timedelta(seconds=float(dt))
    return [
        dtime + (dt_td * i)
        for i in range(counts)
    ]

