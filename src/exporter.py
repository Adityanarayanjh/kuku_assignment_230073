"""Writers for JSON / CSV / Markdown outputs."""
import csv
import json
from pathlib import Path

from .timeutil import sec_to_ts


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def seg_text(p):
    return "; ".join(f"Ep{s['episode']} {sec_to_ts(s['start'])}-{sec_to_ts(s['end'])}" for s in p["segments"])


def promos_to_rows(promos):
    rows = []
    for p in promos:
        sc = p["scores"]
        rows.append({
            "rank": p.get("rank", ""), "promo_id": p["promo_id"], "decision": p["decision"],
            "episodes": ",".join(str(e) for e in sorted({s["episode"] for s in p["segments"]})),
            "segments": seg_text(p), "total_duration_s": round(p["total_duration"], 2),
            "final_score": p["final_score"], "hook_score": sc["hook_score"], "cliffhanger_score": sc["cliffhanger_score"],
            "clarity_score": sc["standalone_clarity_score"], "emotion_score": sc["emotional_impact_score"],
            "coherence_score": sc["narrative_coherence_score"], "promo_potential_score": sc["promotional_potential_score"],
            "hook": p["hook"], "cliffhanger": p["cliffhanger"], "why_it_works": p["why_it_works"],
            "hook_type": p.get("hook_type", ""), "characters": ", ".join(p.get("characters", [])),
            "reliability": p["reliability"]["level"], "weaknesses": " | ".join(p.get("weaknesses", [])),
            "engine": p["engine"],
        })
    return rows


def write_csv(path, promos):
    rows = promos_to_rows(promos)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)


def editorial_markdown(manifest, ranked, review, best_by_ep, limit=10) -> str:
    L = [f"# PromoForge editorial report - {manifest['show']}", "",
         f"**Engine:** {manifest['engine_label']}  ", f"**Episodes processed:** {manifest['episodes']}  ",
         f"**Accepted:** {manifest['accepted']} | **Needs review:** {manifest['review']} | **Rejected:** {manifest['rejected']}", "",
         "> Screenplay-based recommendations. Timestamps are script timings, not frame-accurate; an editor must "
         "watch the footage before publishing. No engagement prediction is made.", "", f"## Top {limit} promos", ""]
    for p in ranked[:limit]:
        L += [f"### #{p['rank']} - {seg_text(p)} ({p['total_duration']:.1f}s) - score {p['final_score']:.2f}",
              f"- **Hook:** {p['hook']}", f"- **Cliffhanger:** {p['cliffhanger']}", f"- **Why it works:** {p['why_it_works']}",
              f"- **Scores:** " + ", ".join(f"{k.replace('_score','')} {v}" for k, v in p["scores"].items()),
              f"- **Reliability:** {p['reliability']['level']} ({'; '.join(p['reliability']['reasons'][:2])})",
              f"- **Weaknesses:** {'; '.join(p['weaknesses']) or 'none flagged'}", ""]
    L += ["## Best per episode", "", "| Episode | Best promo | Score | Status |", "|---|---|---|---|"]
    for ep, p in sorted(best_by_ep.items()):
        L.append(f"| {ep} | {seg_text(p)} | {p['final_score']:.2f} | {p['decision']} |" if p else f"| {ep} | none | - | no strong moment |")
    return "\n".join(L)
