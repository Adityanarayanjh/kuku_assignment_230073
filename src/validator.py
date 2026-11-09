"""Independent checks of every promo against the PARSED SOURCE (not against the generator's own data).
A promo that fails a hard check can never be 'accept'."""
import config
from .timeutil import sec_to_ts

TOL = 0.011   # rounding tolerance (timestamps have 0.01 s resolution)
PRECISION_NOTE = ("Times come from the screenplay (lines/actions are timed to 0.01 s). They are script timings, "
                  "not frame-accurate video cut points: confirm against the actual footage before publishing.")


def _norm(t):
    return " ".join(str(t).split()).strip('"“” ').lower()


def validate_promo(promo: dict, episodes_by_id: dict) -> dict:
    checks, failures, warnings = [], [], []

    def chk(name, ok, detail="", hard=True):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            (failures if hard else warnings).append(f"{name}: {detail}")

    segs = promo.get("segments", [])
    chk("has_segments", bool(segs), "no segments")
    total, prev_end, prev_ep, source_events = 0.0, None, None, []
    for i, sg in enumerate(segs, 1):
        ep = episodes_by_id.get(sg.get("episode"))
        chk(f"seg{i}_episode_exists", ep is not None, f"episode {sg.get('episode')} not in input")
        if ep is None:
            continue
        s, e = sg["start"], sg["end"]
        chk(f"seg{i}_start_before_end", s < e, f"{s} >= {e}")
        chk(f"seg{i}_inside_episode", 0 - TOL <= s and e <= ep.duration + TOL,
            f"{sec_to_ts(s)}-{sec_to_ts(e)} vs episode length {sec_to_ts(ep.duration)}")
        chk(f"seg{i}_duration_matches", abs((e - s) - sg.get("duration", -1)) <= TOL,
            f"stored {sg.get('duration')} vs computed {round(e - s, 3)}")
        if prev_end is not None and sg["episode"] == prev_ep:
            chk(f"seg{i}_chronological", s >= prev_end - TOL, "segments overlap / out of order")
        if prev_ep is not None and sg["episode"] < prev_ep:
            chk(f"seg{i}_episode_order", False, "episodes out of order")
        prev_end, prev_ep = e, sg["episode"]
        total += (e - s)
        evs = [x for x in ep.events() if x.start is not None and x.end is not None
               and x.start >= s - TOL and x.start < e - TOL and x.kind in ("DIALOGUE", "ACTION")]
        chk(f"seg{i}_has_source_content", len(evs) > 0, "no screenplay dialogue/action in this range")
        source_events += evs
        # cuts must not slice a spoken line
        for t, lab in ((s, "start"), (e, "end")):
            cut = [x for x in ep.events() if x.kind == "DIALOGUE" and x.start is not None
                   and x.start + 0.05 < t < x.end - 0.05]
            chk(f"seg{i}_{lab}_not_mid_line", not cut,
                f"cuts through a line at {sec_to_ts(t)}", hard=False)
        # source-data uncertainty flags overlapping this segment
        ids = {x.event_id for x in evs}
        bad = [f for f in ep.flags if f["type"].startswith("event_") and f["detail"] in ids]
        chk(f"seg{i}_source_timestamps_clean", not bad, f"source flags: {[b['type'] for b in bad]}", hard=False)
    chk("total_duration_matches", abs(total - promo.get("total_duration", -1)) <= TOL * max(1, len(segs)),
        f"stored {promo.get('total_duration')} vs computed {round(total, 3)}")
    chk("min_duration_20s", total >= config.MIN_DURATION - TOL, f"{total:.2f}s")
    chk("max_duration_60s", total <= config.MAX_DURATION + TOL, f"{total:.2f}s")
    if len({sg.get("episode") for sg in segs}) > 1:
        warnings.append("cross_episode: combination needs editor coherence review")

    # evidence traceability: every quoted evidence/hook/cliffhanger string must exist verbatim in source
    corpus = [_norm(x.text) for x in source_events]
    unsupported = []
    for item in promo.get("evidence", []):
        q = _norm(item.get("quote", ""))
        if q and not any(q in c for c in corpus):
            unsupported.append(item.get("quote", "")[:60])
    chk("evidence_traceable", not unsupported, f"quotes not found in the clip's source text: {unsupported}")
    return {"passed": not failures, "failures": failures, "warnings": warnings, "checks": checks,
            "precision_note": PRECISION_NOTE}


def reliability(promo: dict, validation: dict) -> dict:
    """Reliability of the RECOMMENDATION (not the model's self-confidence)."""
    reasons, level = [], 3
    if not validation["passed"]:
        return {"level": "Low", "reasons": ["failed validation: " + "; ".join(validation["failures"][:2])]}
    if validation["warnings"]:
        level -= 1; reasons.append(f"{len(validation['warnings'])} validation warning(s)")
    sc = promo["scores"]
    if sc["standalone_clarity_score"] < 6:
        level -= 1; reasons.append("clarity below 6: new viewers may be lost")
    if promo.get("engine", "").startswith("heuristic"):
        level = min(level, 2); reasons.append("scored by rules, not by story understanding")
    if len(promo["segments"]) > 1:
        level -= 1; reasons.append("multi-segment cut")
    if promo["final_score"] < 6.5:
        level -= 1; reasons.append("low margin over accept threshold")
    reasons.append("script timings only; footage check required")
    return {"level": {3: "High", 2: "Medium"}.get(level, "Low"), "reasons": reasons}
