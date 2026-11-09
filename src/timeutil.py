"""Timestamp helpers. Screenplay timestamps look like 01:23.45 (mm:ss.cc)."""
import re

_TS = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$|^(\d+(?:\.\d+)?)$")


def ts_to_sec(ts: str) -> float:
    """'01:23.45' -> 83.45 ; also accepts 'hh:mm:ss.cc' and plain seconds."""
    ts = ts.strip()
    m = _TS.match(ts)
    if not m:
        raise ValueError(f"Malformed timestamp: {ts!r}")
    if m.group(4) is not None:
        return float(m.group(4))
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2))
    seconds = float(m.group(3))
    if seconds >= 60 or (m.group(1) and minutes >= 60):
        raise ValueError(f"Out-of-range timestamp: {ts!r}")
    return round(hours * 3600 + minutes * 60 + seconds, 3)


def sec_to_ts(sec: float) -> str:
    """83.45 -> '01:23.45' (rounded to 0.01 s, with correct carry)."""
    cs = int(round(sec * 100))
    minutes, rem = divmod(cs, 6000)
    return f"{minutes:02d}:{rem // 100:02d}.{rem % 100:02d}"


def duration(start: float, end: float) -> float:
    if end < start:
        raise ValueError("end before start")
    return round(end - start, 3)
