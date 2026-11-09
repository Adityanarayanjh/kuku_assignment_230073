"""Stage C: context-aware evaluation.

Two interchangeable evaluators return the SAME dict shape (see schema.SCORE_KEYS):
  * evaluate_segments_heuristic  - deterministic rules ("offline engine"; NOT an LLM)
  * evaluate_with_llm            - sends the candidate excerpt to an LLM (needs API key)
Both produce 0-10 scores + evidence quotes that must exist verbatim in the screenplay.
"""
import json
import re
from pathlib import Path

import config
from . import lexicon as lx
from .schema import validate_evaluation, SchemaError
from .timeutil import sec_to_ts
from .window import Timeline

HOOK_LABEL = {"betrayal": "betrayal/humiliation", "secret": "secret/reveal", "threat": "threat/danger",
              "violence": "violence/physical", "romance": "romance", "status": "power/status",
              "shock": "shock/question", "ultimatum": "ultimatum/confrontation"}
ENTRANCE = re.compile(r"\b(enters?|emerges?|appears?|bursts?|steps? out|opens?|arrives?|stands? (?:in|at) the doorway|"
                      r"reveals?|unveils?|discovers?|finds?)\b", re.I)
FX_TENSE = re.compile(r"(cliffhanger|ominous|unresolved|tension|pounding|explosion|detonation|crash|thud|slam)", re.I)


def clamp(x, lo=0.0, hi=10.0):
    return max(lo, min(hi, x))


def _int(tl: Timeline, ev):
    cache = tl.__dict__.setdefault("_int", {})
    if ev.event_id not in cache:
        cache[ev.event_id] = lx.intensity(ev.kind, ev.text, ev.tone)
    return cache[ev.event_id]


def final_score(scores: dict) -> float:
    """Weighted score + hard gate: a weak hook or weak cliffhanger caps the result."""
    f = sum(config.WEIGHTS[k] * scores[k] for k in config.WEIGHTS)
    if min(scores["hook_score"], scores["cliffhanger_score"]) < config.GATE_SOFT:
        f = min(f, config.GATE_CAP)
    return round(f, 3)


def evaluate_segments_heuristic(tl: Timeline, segments, scene_lookup=None) -> dict:
    """segments: list of (start, end) in ONE episode, chronological. Returns an evaluation dict."""
    ep = tl.ep
    s0, e_last = segments[0][0], segments[-1][1]
    total = sum(e - s for s, e in segments)
    inside, seen = [], set()
    for s, e in segments:
        for ev in tl.content_inside(s, e):
            if ev.event_id not in seen:
                seen.add(ev.event_id); inside.append(ev)
    ints = [(ev, *_int(tl, ev)) for ev in inside]
    evidence, weaknesses, issues = [], [], []

    # ---------- hook: first 3 seconds of the clip ----------
    first3 = [t for t in ints if t[0].start < s0 + config.HOOK_WINDOW]
    next8 = [t for t in ints if s0 + config.HOOK_WINDOW <= t[0].start < s0 + 8]
    max3 = max([t[1] for t in first3], default=0.0)
    max8 = max([t[1] for t in next8], default=0.0)
    hook = 2.0 + 0.85 * max3 + 0.15 * max8
    first = inside[0] if inside else None
    if not first3:
        hook -= 2.0; weaknesses.append("nothing happens in the first 3 seconds")
    opens_mid = bool(first and first.kind == "DIALOGUE" and lx.CONNECTIVE_START.match(first.text))
    if opens_mid:
        hook -= 1.5; issues.append(f'opens mid-conversation ("{first.text[:40]}...")')
    hook = clamp(hook)
    hook_ev = max(first3, key=lambda t: t[1])[0] if first3 else first
    if hook_ev:
        evidence.append({"dimension": "hook", "event_id": hook_ev.event_id, "timestamp": sec_to_ts(hook_ev.start),
                         "quote": hook_ev.text[:200]})

    # ---------- cliffhanger: closing beat ----------
    closing = [t for t in ints if t[0].end >= e_last - 3.0]
    last = max(inside, key=lambda x: x.end) if inside else None
    cliff = 2.5
    reasons = []
    if last:
        txt = last.text.strip()
        if last.kind == "DIALOGUE" and txt.endswith("?"):
            cliff += 2.0; reasons.append("ends on an unanswered question")
            nxt = tl.next_content_after(e_last)
            if nxt and nxt.start - e_last < 4.0:
                cliff += 1.0; reasons.append("cut right before the reply")
        if txt.endswith("..."):
            cliff += 1.0
        cliff += min(3.5, 0.7 * max([t[1] for t in closing], default=0.0))
        if ENTRANCE.search(txt) and last.kind == "ACTION":
            cliff += 1.0; reasons.append("ends on an entrance/discovery")
        if lx.RESOLVING.search(txt):
            cliff -= 3.0; weaknesses.append("final line sounds like a resolution")
        if e_last >= ep.duration - 1.5:
            cliff += 2.0; reasons.append("episode's own closing beat")
            fx = [x for x in tl.events if x.kind == "SOUND_FX" and x.end >= ep.duration - 3]
            if fx and FX_TENSE.search(fx[-1].text):
                cliff += 0.5
        elif tl.next_content_after(e_last) is None:
            weaknesses.append("nothing follows in the screenplay (check ending)")
        evidence.append({"dimension": "cliffhanger", "event_id": last.event_id,
                         "timestamp": sec_to_ts(last.end), "quote": txt[:200]})
    cliff = clamp(cliff)

    # ---------- standalone clarity ----------
    clarity = 6.5
    speakers = [t[0].speaker for t in ints if t[0].speaker]
    distinct = list(dict.fromkeys(speakers))
    text_all = " ".join(ev.text for ev in inside)
    if lx.RELATION.search(text_all) or len(distinct) >= 2:
        clarity += 1.0
    if s0 <= 0.5 and ep.episode_id == 1:
        clarity += 1.5                      # series opener: nothing to be missing
    elif s0 <= 0.5:
        clarity += 0.5
    if first and first.kind == "DIALOGUE" and lx.PRONOUN_START.match(first.text):
        clarity -= 1.5; issues.append("first line is a pronoun with no named referent")
    if opens_mid:
        clarity -= 2.0
    if len(distinct) > 4:
        clarity -= 1.0; issues.append(f"{len(distinct)} different speakers (hard to follow)")
    if s0 <= 5 and ep.episode_id > 1 and re.search(r"picks up (immediately|directly)", ep.continuity, re.I):
        clarity -= 1.0; issues.append("episode continues directly from the previous one")
    # named people mentioned in dialogue who never speak in the clip = unexplained references
    spoken_names = {d.split()[-1].lower() for d in distinct}
    unseen = set()
    for ev in inside:
        if ev.kind == "DIALOGUE":
            for m in lx.PROPER_NOUN.finditer(ev.text):
                w = m.group(1)
                if w.lower() not in spoken_names and w.lower() not in {"you", "your", "god", "lord", "lady"}:
                    unseen.add(w)
    if unseen:
        clarity -= min(2.0, 0.5 * len(unseen))
        issues.append("names mentioned but not introduced in clip: " + ", ".join(sorted(unseen)[:4]))
    clarity = clamp(clarity)
    if first:
        evidence.append({"dimension": "standalone_clarity", "event_id": first.event_id,
                         "timestamp": sec_to_ts(first.start), "quote": first.text[:200]})

    # ---------- emotional impact ----------
    vals = sorted([t[1] for t in ints], reverse=True)
    top4 = vals[:4]
    emo = clamp(1.5 + 0.95 * (sum(top4) / len(top4) if top4 else 0) + 0.15 * sum(v >= 4 for v in vals))
    peak = max(ints, key=lambda t: t[1])[0] if ints else None
    if peak:
        evidence.append({"dimension": "emotional_impact", "event_id": peak.event_id,
                         "timestamp": sec_to_ts(peak.start), "quote": peak.text[:200]})

    # ---------- narrative coherence ----------
    coh = 8.5
    scenes_used = {ev.scene_id for ev in inside}
    if len(scenes_used) > 1:
        coh -= 1.5; weaknesses.append("clip crosses a scene change")
    if len(segments) > 1:
        coh -= 1.0; weaknesses.append("multi-segment clip: jump cut needs editor review")
    if len(distinct) > 3:
        coh -= 0.7 * (len(distinct) - 3)
    spoken_secs = sum(ev.end - ev.start for ev in inside if ev.kind == "DIALOGUE")
    if total and spoken_secs / total < 0.35:
        coh -= 1.0; weaknesses.append("little dialogue (screenplay gives mostly visual action)")
    gaps = 0
    for a, b in zip(inside, inside[1:]):
        if b.start - a.end > 5.0:
            gaps += 1
    if gaps:
        coh -= 0.7 * gaps; weaknesses.append(f"{gaps} silent stretch(es) > 5s")
    coh = clamp(coh)

    # ---------- promotional potential ----------
    cats = {}
    for ev, sc, hits in ints:
        for h in hits:
            cats[h] = cats.get(h, 0) + 1
    core = [c for c in cats if c != "ultimatum"]
    turns = sum(1 for a, b in zip(speakers, speakers[1:]) if a != b)
    promo = clamp(2.5 + 1.0 * min(len(core), 4) + 0.35 * max(vals[:1] or [0]) + (0.5 if turns >= 3 else 0))

    # hook type from the first ~10 s
    early = {}
    for ev, sc, hits in ints:
        if ev.start < s0 + 10:
            for h in hits:
                early[h] = early.get(h, 0) + 1
    hook_type = HOOK_LABEL[max(early, key=early.get)] if early else "dialogue tension"

    if len(distinct) >= 1 and not [t for t in ints if t[1] >= 3]:
        weaknesses.append("no high-intensity line detected (low emotional peak)")

    scores = {"hook_score": hook, "cliffhanger_score": cliff, "standalone_clarity_score": clarity,
              "emotional_impact_score": emo, "narrative_coherence_score": coh,
              "promotional_potential_score": promo}
    scores = {k: round(v, 2) for k, v in scores.items()}

    # ---------- narrative progression (extractive) ----------
    def who(ev): return " ".join(w.capitalize() for w in ev.speaker.split()) if ev.speaker else "Scene"
    mid = [t for t in ints if s0 + total * 0.3 <= t[0].start <= s0 + total * 0.8]
    dev = max(mid, key=lambda t: t[1])[0] if mid else (inside[len(inside) // 2] if inside else None)
    narrative = {
        "hook": f'{who(hook_ev)}: "{hook_ev.text[:120]}"' if hook_ev else "",
        "development": f'{who(dev)}: "{dev.text[:120]}"' if dev else "",
        "reveal": f'{who(peak)}: "{peak.text[:120]}"' if peak else "",
        "open_question": f'Ends on {who(last)}: "{last.text[:120]}"' if last else "",
        "why_continue": "; ".join(reasons) if reasons else "ends mid-conflict with no on-screen resolution",
        "method": "extractive (heuristic engine) - quotes from screenplay, no interpretation",
    }
    context = {"level": "self_contained" if clarity >= 7 else ("needs_context" if clarity >= 5 else "unclear"),
               "depends_on_prior_episodes": bool(opens_mid or (clarity < 5 and ep.episode_id > 1)),
               "issues": issues, "bridge_suggestion": None}
    return {
        "scores": scores, "final_score": final_score(scores), "evidence": evidence,
        "weaknesses": weaknesses, "hook_type": hook_type, "characters": distinct,
        "hook": narrative["hook"], "cliffhanger": narrative["open_question"],
        "why_it_works": (f"Opens on {hook_type} and " + (reasons[0] if reasons else "stops before any resolution")
                         + f"; {len(distinct)} speaker(s), {total:.0f}s."),
        "narrative": narrative, "context_check": context,
        "event_ids": [ev.event_id for ev in inside], "categories": cats,
    }


# ------------------------------ LLM evaluation ------------------------------
def _excerpt(tl: Timeline, segments, with_camera=False):
    lines = []
    for i, (s, e) in enumerate(segments, 1):
        lines.append(f"[segment {i}: {sec_to_ts(s)} - {sec_to_ts(e)}]")
        for ev in tl.inside(s, e):
            if ev.kind == "CAMERA" and not with_camera:
                continue
            who = ev.speaker or ev.kind
            lines.append(f"{sec_to_ts(ev.start)} {who}: {ev.text}")
    return "\n".join(lines)


def evaluate_with_llm(client, tl: Timeline, segments, show_name="", prior_context="") -> dict:
    """Ask the LLM to score a candidate. Raises SchemaError if the answer is unusable after retries."""
    prompt = Path(config.ROOT / "prompts" / "evaluation_prompt.txt").read_text(encoding="utf-8")
    ep = tl.ep
    user = (prompt.replace("{{SHOW}}", show_name).replace("{{EPISODE}}", str(ep.episode_id))
            .replace("{{LOGLINE}}", ep.logline).replace("{{CONTINUITY}}", ep.continuity)
            .replace("{{PRIOR_CONTEXT}}", prior_context or "(none provided)")
            .replace("{{EXCERPT}}", _excerpt(tl, segments)))
    raw = client.generate_json(user, validator=validate_evaluation)
    ev = dict(raw)
    scores = {k: ev.pop(k) for k in list(ev) if k.endswith("_score")}
    ev["scores"] = scores
    ev["final_score"] = final_score(scores)
    return ev
