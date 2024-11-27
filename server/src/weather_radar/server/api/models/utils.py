from datetime import datetime

time_t = float|str|None

def datetime_from_time(time: time_t) -> datetime:
    if time is None:
        return datetime.now()
    try:
        dtime = float(time)
    except ValueError:
        return datetime.fromisoformat(str(time))
    else:
        return datetime.fromtimestamp(dtime)
