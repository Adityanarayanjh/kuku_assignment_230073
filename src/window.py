"""Helpers for reasoning about a candidate window [start, end] inside one episode."""
from .schema import Episode

EPS = 0.05


class Timeline:
    def __init__(self, ep: Episode):
        self.ep = ep
        self.events = sorted([e for e in ep.events() if e.start is not None and e.end is not None],
                             key=lambda e: (e.start, e.end))
        self.dialogue = [e for e in self.events if e.kind == "DIALOGUE"]
        self.content = [e for e in self.events if e.kind in ("DIALOGUE", "ACTION")]
        self.duration = ep.duration

    def straddles_dialogue(self, t: float) -> bool:
        """True if cutting at t would slice through a spoken line."""
        return any(d.start + EPS < t < d.end - EPS for d in self.dialogue)

    def valid_starts(self):
        out = {0.0} if self.content else set()
        for e in self.content:
            out.add(e.start)
        return sorted(t for t in out if not self.straddles_dialogue(t))

    def valid_ends(self):
        out = {self.duration}
        for e in self.events:
            if e.kind in ("DIALOGUE", "ACTION", "SOUND_FX"):
                out.add(e.end)
        return sorted(t for t in out if not self.straddles_dialogue(t))

    def inside(self, s: float, e: float):
        """Events that begin inside the window."""
        return [x for x in self.events if x.start >= s - EPS and x.start < e - EPS]

    def content_inside(self, s, e):
        return [x for x in self.content if x.start >= s - EPS and x.start < e - EPS]

    def next_content_after(self, t):
        for x in self.content:
            if x.start >= t - EPS:
                return x
        return None

    def prev_content_before(self, t, n=1):
        prev = [x for x in self.content if x.end <= t + EPS]
        return prev[-n:] if prev else []
