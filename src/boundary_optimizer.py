"""Stage D/E: find the best start & end for each seed, then (optionally) add a context bridge.

For a seed we try every legal (start, end) pair:
  * start = beginning of a spoken line or action (never mid-line),
  * end   = end of a line/action/sound cue (never mid-line),
  * 20 s <= end-start <= 60 s, and the seed's peak moment stays inside the window.
The pair with the best final score wins. Candidates then go through NMS (drop overlaps).
"""
import config
from .evaluator import evaluate_segments_heuristic, final_score, _int
from .window import Timeline
from . import lexicon as lx


def best_window(tl: Timeline, seed):
    if seed.get("fixed_window"):
        s, e = seed["fixed_window"]
        return (s, e), evaluate_segments_heuristic(tl, [(s, e)])
    starts, ends = tl.valid_starts(), tl.valid_ends()
    t0, t1 = seed["time"], seed["peak_end"]
    if seed["type"] == "episode_open":
        starts = [0.0] if 0.0 in starts else starts[:1]
    best = None
    for s in starts:
        if seed["type"] == "peak" and not (t0 - 45 <= s <= t0 + 1e-6):
            continue
        if seed["type"] == "episode_end" and s > tl.duration - config.MIN_DURATION:
            continue
        if seed["type"] == "episode_end" and s < tl.duration - config.MAX_DURATION:
            continue
        for e in ends:
            d = e - s
            if d < config.MIN_DURATION - 1e-6 or d > config.MAX_DURATION + 1e-6:
                continue
            if seed["type"] == "peak" and e < t1 - 1e-6:
                continue
            if seed["type"] == "episode_end" and e < tl.duration - 1.5:
                continue
            ev = evaluate_segments_heuristic(tl, [(s, e)])
            key = (ev["final_score"], -d)          # tie-break: shorter is punchier
            if best is None or key > best[0]:
                best = (key, (s, e), ev)
    if best is None:
        return None, None
    return best[1], best[2]


def overlap_ratio(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))


def nms(cands, thresh=0.5):
    """cands: list of dicts with 'window' and 'final_score'. Returns (kept, dropped)."""
    kept, dropped = [], []
    for c in sorted(cands, key=lambda c: -c["final_score"]):
        dup = next((k for k in kept if overlap_ratio(k["window"], c["window"]) > thresh), None)
        if dup:
            c = dict(c); c["dropped_for"] = dup["candidate_id"]
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


def try_bridge(tl: Timeline, window, base_eval):
    """Feature 2 (context checker, fix side): if the clip is hard to follow for a new viewer, look for a
    short EARLIER segment (<= 8 s, same episode) that introduces the people/relationship and still
    hits within 3 s. Returns a bridged candidate dict, or None."""
    s, e = window
    if base_eval["scores"]["standalone_clarity_score"] >= 7.0 or s < 4 or (e - s) > config.MAX_DURATION - 6:
        return None
    best = None
    for bs in tl.valid_starts():
        if not (s - 40 <= bs <= s - 5):
            continue
        for be in tl.valid_ends():
            d = be - bs
            if d < 3 or d > 8 or be > s - 1.0:
                continue
            evs = tl.content_inside(bs, be)
            if not evs:
                continue
            txt = " ".join(x.text for x in evs)
            first_int = max((_int(tl, x)[0] for x in evs if x.start < bs + config.HOOK_WINDOW), default=0)
            if first_int < 3.0 or not lx.RELATION.search(txt):
                continue
            if (e - s) + d > config.MAX_DURATION:
                continue
            ev = evaluate_segments_heuristic(tl, [(bs, be), (s, e)])
            if best is None or ev["final_score"] > best[2]["final_score"]:
                best = (bs, be, ev)
    if best is None:
        return None
    bs, be, ev = best
    gain = ev["final_score"] - base_eval["final_score"]
    return {"segments": [(bs, be), (s, e)], "eval": ev, "gain": round(gain, 3)}
