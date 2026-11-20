"""Resolve a stream request's frame epochs.

A request names the frames it wants in up to two ways, which may be combined:

* ``times`` — a comma-separated list, streamed first, in the order given. Each
  item is either one epoch (``1795547700``) or a RUN ``from:to:step`` — every
  epoch from ``from`` to ``to`` inclusive, ``step`` seconds apart, walking in
  whichever direction ``to`` lies (``1795547700:1795330800:900`` counts DOWN).
  Runs keep the URL short: a client filling hundreds of consecutive gap frames
  sends one ~30-character run instead of hundreds of numbers (a long list can
  exceed a proxy's request-line limit and silently fail the fill).
* ``start``/``end``/``dt`` — a contiguous sequence, streamed after the batch.

Duplicates are dropped, keeping the first occurrence, so a frame named in both
is computed/sent once. Malformed items are ignored. The total is capped at
MAX_FRAMES so a pathological request can't generate unbounded work.
"""

MAX_FRAMES = 5000


def frame_times(start, end, dt, times, limit=MAX_FRAMES):
    out = []
    seen = set()

    def add(epoch):
        if len(out) >= limit:
            return False
        e = int(epoch)
        if e not in seen:
            seen.add(e)
            out.append(e)
        return True

    # Batch first, in the order given.
    if times:
        for tok in str(times).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                if ":" in tok:
                    a, b, s = tok.split(":")
                    a, b, step = float(a), float(b), abs(float(s))
                    if step <= 0:
                        continue
                    direction = 1 if b >= a else -1
                    for i in range(int(abs(b - a) // step) + 1):
                        if not add(a + direction * step * i):
                            break
                else:
                    add(float(tok))
            except (ValueError, OverflowError):
                continue

    # Then the contiguous sequence.
    if start is not None and end is not None:
        step = int(dt) if dt and int(dt) > 0 else 3600
        t = float(start)
        while t <= float(end) + 1:
            if not add(t):
                break
            t += step

    return out
