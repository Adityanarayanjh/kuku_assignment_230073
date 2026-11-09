"""Runs all stages end to end and writes every output file."""
import json
import time
from collections import Counter
from pathlib import Path

import config
from . import candidate_extractor as ce
from . import boundary_optimizer as bo
from .diversity import rank_diverse
from .evaluator import evaluate_segments_heuristic, evaluate_with_llm, final_score, _excerpt
from .exporter import write_json, write_csv, editorial_markdown
from .llm_client import LLMError
from .ranker import decide
from .schema import to_dict
from .timeutil import sec_to_ts
from .validator import validate_promo, reliability
from .window import Timeline

HEUR_LABEL = "heuristic-offline (rule-based scoring, NO LLM)"


def _segment_dicts(tl, segments):
    out = []
    for s, e in segments:
        evs = tl.content_inside(s, e)
        out.append({"episode": tl.ep.episode_id, "start": round(s, 3), "end": round(e, 3),
                    "start_ts": sec_to_ts(s), "end_ts": sec_to_ts(e), "duration": round(e - s, 3),
                    "scene_ids": sorted({x.scene_id for x in evs}), "event_ids": [x.event_id for x in evs],
                    "source_lines": [min(x.line_start for x in evs), max(x.line_end for x in evs)] if evs else None})
    return out


def _excerpt_rows(tl, segments):
    rows = []
    for s, e in segments:
        for ev in tl.content_inside(s, e):
            rows.append({"episode": tl.ep.episode_id, "t": sec_to_ts(ev.start), "kind": ev.kind,
                         "speaker": ev.speaker, "tone": ev.tone, "text": ev.text, "event_id": ev.event_id})
    return rows


def make_promo(tl, segments, ev, engine_label, cid, extra=None):
    segs = _segment_dicts(tl, segments)
    p = {
        "promo_id": cid, "engine": engine_label, "segments": segs,
        "total_duration": round(sum(s["duration"] for s in segs), 3),
        "scores": ev["scores"], "final_score": ev["final_score"],
        "hook": ev.get("hook", ""), "cliffhanger": ev.get("cliffhanger", ""),
        "why_it_works": ev.get("why_it_works", ""), "narrative": ev.get("narrative", {}),
        "evidence": ev.get("evidence", []), "weaknesses": ev.get("weaknesses", []),
        "context_check": ev.get("context_check", {}), "hook_type": ev.get("hook_type", ""),
        "characters": ev.get("characters", []), "excerpt": _excerpt_rows(tl, segments),
        "episode_logline": tl.ep.logline,
    }
    p.update(extra or {})
    return p


def run_pipeline(show, out_dir, engine="heuristic", client=None, episode_filter=None, llm_eval_top=3, log=print):
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = []

    def L(msg):
        logs.append(msg); log(msg)

    engine_label = HEUR_LABEL if engine == "heuristic" else f"llm:{getattr(client, 'model', '?')} (+rule prefilter)"
    episodes = [e for e in show.episodes if not episode_filter or e.episode_id in episode_filter]
    by_id = {e.episode_id: e for e in episodes}
    L(f"[A] parsed {len(episodes)} episodes, {sum(len(e.scenes) for e in episodes)} scenes, "
      f"{sum(len(e.events()) for e in episodes)} events; engine = {engine_label}")
    write_json(out / "parsed_episodes.json", {"show": show.name, "show_flags": show.flags,
                                              "episodes": [to_dict(e) for e in episodes]})

    all_seeds, candidates, rejected, model_errors = [], [], [], []
    for ep in episodes:
        tl = Timeline(ep)
        seeds = ce.find_seeds(tl)
        if engine == "llm":
            try:
                seeds += ce.discover_with_llm(client, tl, show.name)
            except LLMError as exc:
                model_errors.append({"episode": ep.episode_id, "stage": "discovery", "error": str(exc)})
                L(f"[B] ep{ep.episode_id}: LLM discovery failed ({exc}); continuing with rule seeds")
        all_seeds += seeds
        ep_cands = []
        for sd in seeds:
            win, ev = bo.best_window(tl, sd)
            if win is None:
                rejected.append({"candidate_id": sd["seed_id"], "episode": ep.episode_id, "window": None,
                                 "reasons": ["no legal 20-60s window around this seed"], "stage": "boundary"})
                continue
            ep_cands.append({"seed": sd, "window": win, "final_score": ev["final_score"], "eval": ev,
                             "candidate_id": sd["seed_id"].replace("SEED", "C").replace("LLM", "L")})
        if engine == "llm":
            # LLM-proposed windows must reach the LLM judge; only dedupe rule windows vs rule windows here
            rule_c = [c for c in ep_cands if c["seed"]["source"] == "rules"]
            llm_c = [c for c in ep_cands if c["seed"]["source"] == "llm"]
            k1, d1 = bo.nms(rule_c)
            k2, d2 = bo.nms(llm_c)
            kept, dropped = k1 + k2, d1 + d2
        else:
            kept, dropped = bo.nms(ep_cands)
        for d in dropped:
            rejected.append({"candidate_id": d["candidate_id"], "episode": ep.episode_id,
                             "window": [d["window"][0], d["window"][1]], "final_score": d["final_score"],
                             "reasons": [f"duplicate: overlaps stronger candidate {d['dropped_for']}"], "stage": "dedupe"})
        for d in kept[config.PER_EPISODE_CANDIDATES:]:
            rejected.append({"candidate_id": d["candidate_id"], "episode": ep.episode_id,
                             "window": [d["window"][0], d["window"][1]], "final_score": d["final_score"],
                             "reasons": [f"below per-episode cap of {config.PER_EPISODE_CANDIDATES}"], "stage": "cap"})
        for c in kept[:config.PER_EPISODE_CANDIDATES]:
            c["tl"] = tl
            candidates.append(c)
    L(f"[B/D] {len(all_seeds)} seeds -> {len(candidates)} distinct candidates after boundary search + dedupe "
      f"({len(rejected)} dropped)")
    write_json(out / "candidates.json", {
        "seeds": all_seeds,
        "candidates": [{"candidate_id": c["candidate_id"], "episode": c["seed"]["episode"],
                        "window": [c["window"][0], c["window"][1]], "window_ts": [sec_to_ts(c["window"][0]), sec_to_ts(c["window"][1])],
                        "seed_type": c["seed"]["type"], "seed_source": c["seed"]["source"],
                        "prefilter_score": c["final_score"]} for c in candidates]})

    # ---- Stage C/E: evaluation (+ bridge) ----
    promos, raw_evals = [], []
    llm_budget = Counter()
    for c in sorted(candidates, key=lambda c: -c["final_score"]):
        tl, ep_id = c["tl"], c["seed"]["episode"]
        segments, ev, extra = [c["window"]], c["eval"], {"seed": {k: c["seed"][k] for k in ("seed_id", "type", "source", "why")}}
        if engine == "llm":
            if llm_budget[ep_id] >= llm_eval_top and c["seed"]["source"] != "llm":
                rejected.append({"candidate_id": c["candidate_id"], "episode": ep_id, "window": list(c["window"]),
                                 "final_score": c["final_score"], "stage": "llm_budget",
                                 "reasons": [f"not sent to LLM: per-episode budget of {llm_eval_top} reached"]})
                continue
            llm_budget[ep_id] += 1
            try:
                ev_llm = evaluate_with_llm(client, tl, segments, show.name)
                extra["heuristic_prefilter"] = {"final_score": ev["final_score"], "scores": ev["scores"]}
                ev_llm.setdefault("hook_type", ev.get("hook_type", "")); ev_llm["characters"] = ev.get("characters", [])
                ev_llm.setdefault("context_check", ev.get("context_check", {}))
                ev = ev_llm
            except LLMError as exc:
                model_errors.append({"candidate_id": c["candidate_id"], "stage": "evaluation", "error": str(exc)})
                rejected.append({"candidate_id": c["candidate_id"], "episode": ep_id, "window": list(c["window"]),
                                 "final_score": c["final_score"], "stage": "model_error", "reasons": [f"LLM evaluation failed: {exc}"]})
                continue
        raw_evals.append({"candidate_id": c["candidate_id"], "engine": engine_label, "window": list(c["window"]), "evaluation": ev})
        br = bo.try_bridge(tl, c["window"], ev) if engine == "heuristic" else None
        if br:
            if br["gain"] >= 0.4:
                extra["unbridged_alternative"] = {"window": list(c["window"]), "final_score": ev["final_score"]}
                segments, ev = br["segments"], br["eval"]
                ev["context_check"]["bridge_suggestion"] = {"applied": True, "gain": br["gain"]}
            else:
                ev["context_check"]["bridge_suggestion"] = {"applied": False, "gain": br["gain"],
                                                            "segments": [[sec_to_ts(a), sec_to_ts(b)] for a, b in br["segments"][:1]],
                                                            "note": "earlier context segment found but did not improve the score enough"}
        promo = make_promo(tl, segments, ev, engine_label, c["candidate_id"], extra)
        promos.append(promo)
    write_json(out / "evaluations_raw.json", raw_evals)
    if engine == "llm":                                   # dedupe across rule + LLM windows using the LLM scores
        keep_ids, cands_by_ep = set(), {}
        for p in promos:
            cands_by_ep.setdefault(p["segments"][0]["episode"], []).append(
                {"candidate_id": p["promo_id"], "window": (p["segments"][0]["start"], p["segments"][-1]["end"]),
                 "final_score": p["final_score"]})
        for ep_id, lst in cands_by_ep.items():
            kept_l, dropped_l = bo.nms(lst)
            keep_ids |= {k["candidate_id"] for k in kept_l}
            for d in dropped_l:
                rejected.append({"candidate_id": d["candidate_id"], "episode": ep_id, "window": list(d["window"]),
                                 "final_score": d["final_score"], "stage": "dedupe",
                                 "reasons": [f"duplicate: overlaps stronger (LLM-scored) candidate {d['dropped_for']}"]})
        promos = [p for p in promos if p["promo_id"] in keep_ids]

    # ---- validation + decisions ----
    val_report, accepted, review, rej_promos = [], [], [], []
    for p in promos:
        v = validate_promo(p, by_id)
        p["validation"] = v
        p["reliability"] = reliability(p, v)
        d = decide(p, v)
        p["decision"], p["decision_reasons"] = d["decision"], d["reasons"]
        val_report.append({"promo_id": p["promo_id"], "passed": v["passed"], "failures": v["failures"], "warnings": v["warnings"]})
        {"accept": accepted, "review": review, "reject": rej_promos}[d["decision"]].append(p)
    for p in rej_promos:
        rejected.append({"candidate_id": p["promo_id"], "episode": p["segments"][0]["episode"],
                         "window": [p["segments"][0]["start"], p["segments"][-1]["end"]], "final_score": p["final_score"],
                         "scores": p["scores"], "reasons": p["decision_reasons"], "stage": "decision",
                         "hook": p["hook"], "cliffhanger": p["cliffhanger"]})
    ranked = rank_diverse(accepted)
    review.sort(key=lambda p: -p["final_score"])
    for i, p in enumerate(review, 1):
        p["rank"] = None; p["review_order"] = i
    best_by_ep = {}
    for ep in episodes:
        pool = [p for p in ranked if p["segments"][0]["episode"] == ep.episode_id] or \
               [p for p in review if p["segments"][0]["episode"] == ep.episode_id]
        best_by_ep[ep.episode_id] = max(pool, key=lambda p: p["final_score"]) if pool else None
    L(f"[C-F] accepted {len(ranked)}, needs review {len(review)}, rejected {len(rejected)}")

    scores = [p["final_score"] for p in ranked]
    manifest = {
        "show": show.name, "engine": engine, "engine_label": engine_label,
        "is_demo_or_offline": engine != "llm", "llm_model": getattr(client, "model", None),
        "run_seconds": round(time.time() - t0, 2), "episodes": len(episodes),
        "scenes": sum(len(e.scenes) for e in episodes), "events": sum(len(e.events()) for e in episodes),
        "seeds": len(all_seeds), "candidates": len(candidates), "accepted": len(ranked), "review": len(review),
        "rejected": len(rejected), "avg_accepted_score": round(sum(scores) / len(scores), 3) if scores else None,
        "episodes_with_accepted_promo": sorted({p["segments"][0]["episode"] for p in ranked}),
        "llm_calls": getattr(client, "calls", 0), "llm_cache_hits": getattr(client, "cache_hits", 0),
        "model_errors": model_errors, "parse_flags": sum(len(e.flags) for e in episodes) + len(show.flags),
        "rejection_reasons": dict(Counter(r["stage"] for r in rejected)),
        "hook_types_accepted": dict(Counter(p["hook_type"] for p in ranked)),
        "weights": config.WEIGHTS, "thresholds": {"hook_min": config.HOOK_MIN, "cliff_min": config.CLIFF_MIN,
                                                 "accept_min_final": config.ACCEPT_MIN_FINAL,
                                                 "clarity_review_below": config.CLARITY_REVIEW_BELOW,
                                                 "diversity_lambda": config.DIVERSITY_LAMBDA},
    }
    write_json(out / "ranked_promos.json", ranked)
    write_csv(out / "ranked_promos.csv", ranked)
    write_json(out / "review_queue.json", review)
    write_json(out / "rejected_candidates.json", rejected)
    write_json(out / "validation_report.json", val_report)
    write_json(out / "best_per_episode.json", {str(k): v for k, v in best_by_ep.items()})
    write_json(out / "evaluation_report.json", {
        "manifest": manifest,
        "parse_flags": [f for e in episodes for f in e.flags] + show.flags,
        "validation_failures": [v for v in val_report if not v["passed"]],
        "validation_warnings": [v for v in val_report if v["warnings"]],
        "low_context_candidates": [p["promo_id"] for p in promos if p["scores"]["standalone_clarity_score"] < config.CLARITY_REVIEW_BELOW],
        "cross_episode": [p["promo_id"] for p in promos if len({s["episode"] for s in p["segments"]}) > 1],
        "bridged": [p["promo_id"] for p in promos if len(p["segments"]) > 1],
        "model_errors": model_errors})
    write_json(out / "run_manifest.json", manifest)
    (out / "editorial_report.md").write_text(editorial_markdown(manifest, ranked, review, best_by_ep), encoding="utf-8")
    (out / "run_log.txt").write_text("\n".join(logs), encoding="utf-8")
    return {"manifest": manifest, "ranked": ranked, "review": review, "rejected": rejected, "best_by_ep": best_by_ep,
            "promos": promos}
