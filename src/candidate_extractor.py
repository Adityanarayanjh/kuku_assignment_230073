"""Stage B: candidate moment discovery.

Cheap deterministic pass: find 'peak' lines (high-intensity dialogue/actions), cluster them,
and also add the opening and the closing beat of each episode as seeds. Each seed is later
refined into exact promo boundaries (Stage D). An optional LLM pass adds seeds whose appeal is
contextual (relationships, reversals) and cannot be seen from keywords.
"""
import config
from . import lexicon as lx
from .evaluator import _int
from .schema import validate_candidates
from .timeutil import ts_to_sec, sec_to_ts
from .window import Timeline

from pathlib import Path

SEED_MIN_INTENSITY = 6.0   # same lines as the old 3.0 before the x2 calibration
CLUSTER_GAP = 10.0


def find_seeds(tl: Timeline, min_intensity=SEED_MIN_INTENSITY, cluster_gap=CLUSTER_GAP):
    ep = tl.ep
    peaks = []
    for ev in tl.content:
        sc, cats = _int(tl, ev)
        if sc >= min_intensity:
            peaks.append((ev, sc, cats))
    # cluster: within cluster_gap seconds keep only the strongest line
    peaks.sort(key=lambda t: t[0].start)
    clusters, cur = [], []
    for p in peaks:
        if cur and p[0].start - cur[-1][0].start > cluster_gap:
            clusters.append(cur); cur = []
        cur.append(p)
    if cur:
        clusters.append(cur)
    seeds = []
    for c in clusters:
        ev, sc, cats = max(c, key=lambda t: t[1])
        seeds.append({"type": "peak", "event_id": ev.event_id, "time": ev.start, "peak_end": ev.end,
                      "intensity": round(sc, 2), "categories": cats, "text": ev.text[:160],
                      "why": f"intensity {sc:.1f}: {', '.join(cats) or 'tone/punctuation'}"})
    if tl.content:
        seeds.append({"type": "episode_open", "event_id": tl.content[0].event_id, "time": 0.0,
                      "peak_end": tl.content[0].end, "intensity": 0.0, "categories": [],
                      "text": tl.content[0].text[:160], "why": "cold open of the episode"})
        seeds.append({"type": "episode_end", "event_id": tl.content[-1].event_id, "time": ep.duration,
                      "peak_end": ep.duration, "intensity": 0.0, "categories": [],
                      "text": tl.content[-1].text[:160], "why": "episode's own closing beat"})
    for i, s in enumerate(seeds, 1):
        s["seed_id"] = f"EP{ep.episode_id:03d}-SEED{i:02d}"
        s["episode"] = ep.episode_id
        s["time_ts"] = sec_to_ts(s["time"])
        s["source"] = "rules"
    return seeds


def transcript_for_llm(ep) -> str:
    """Compact episode text for the LLM: no camera directions, one line per event, timestamps kept."""
    out = [f"EPISODE {ep.episode_id} | duration {sec_to_ts(ep.duration)}", f"Logline: {ep.logline}",
           f"Continuity: {ep.continuity}"]
    for sc in ep.scenes:
        out.append(f"-- SCENE {sc.scene_id}: {sc.heading}")
        for ev in sc.events:
            if ev.kind in ("CAMERA",) or ev.start is None:
                continue
            who = ev.speaker or ev.kind
            tone = f" ({ev.tone})" if ev.tone else ""
            out.append(f"{sec_to_ts(ev.start)}-{sec_to_ts(ev.end)} {who}{tone}: {ev.text}")
    return "\n".join(out)


def discover_with_llm(client, tl: Timeline, show_name=""):
    """Ask the LLM for promo windows in one episode, then SNAP them to real boundaries."""
    prompt = (Path(config.ROOT / "prompts" / "candidate_prompt.txt").read_text(encoding="utf-8")
              .replace("{{SHOW}}", show_name).replace("{{TRANSCRIPT}}", transcript_for_llm(tl.ep)))
    raw = client.generate_json(prompt, validator=validate_candidates)
    items = validate_candidates(raw)
    starts, ends = tl.valid_starts(), tl.valid_ends()
    seeds = []
    for i, c in enumerate(items, 1):
        try:
            s, e = ts_to_sec(str(c["start"])), ts_to_sec(str(c["end"]))
        except ValueError:
            continue
        if not starts or not ends or e <= s:
            continue
        s = min(starts, key=lambda t: abs(t - s))
        cand_e = [t for t in ends if t > s + config.MIN_DURATION - 1e-6 and t <= s + config.MAX_DURATION + 1e-6]
        if not cand_e:
            continue
        e = min(cand_e, key=lambda t: abs(t - e))
        seeds.append({"seed_id": f"EP{tl.ep.episode_id:03d}-LLM{i:02d}", "episode": tl.ep.episode_id,
                      "type": "llm_window", "time": s, "time_ts": sec_to_ts(s), "peak_end": e,
                      "fixed_window": [s, e], "intensity": 0.0, "categories": [],
                      "text": str(c.get("hook_quote", ""))[:160], "why": str(c.get("why", ""))[:300],
                      "source": "llm", "event_id": None})
    return seeds
