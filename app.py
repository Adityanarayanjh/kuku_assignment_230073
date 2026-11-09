"""PromoForge - editor review UI (Streamlit).  Run:  streamlit run app.py
The UI only READS the JSON files written by the pipeline (plus runs the pipeline when you press the button)."""
import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src.exporter import promos_to_rows, editorial_markdown, seg_text

OUT_ROOT = config.ROOT / "outputs"
UPLOAD_ROOT = config.ROOT / "data" / "uploads"
LIMITATIONS = [
    "Works from the screenplay text only: it cannot see acting, framing, music or whether a shot is visually striking.",
    "Timestamps are script timings (lines timed to 0.01 s), not frame-accurate cut points - confirm in the editor.",
    "Offline engine = keyword/rule scoring. It over-rewards '?' and shouting, and misses promos whose power is contextual.",
    "Standalone-clarity is a proxy (name/pronoun/relationship heuristics); only an LLM or a human can judge real comprehension.",
    "Scores are NOT predictions of views, watch-time or conversions. Measure that after posting with real performance data.",
    "Cross-episode combinations are supported by the validator but are only produced in LLM mode and always need editor review.",
]


def _read(path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def list_runs():
    return sorted([d.name for d in OUT_ROOT.glob("run_*") if (d / "run_manifest.json").exists()]) if OUT_ROOT.exists() else []


def load_run(name):
    d = OUT_ROOT / name
    return {"dir": d, "manifest": _read(d / "run_manifest.json", {}), "ranked": _read(d / "ranked_promos.json", []),
            "review": _read(d / "review_queue.json", []), "rejected": _read(d / "rejected_candidates.json", []),
            "report": _read(d / "evaluation_report.json", {}), "best": _read(d / "best_per_episode.json", {}),
            "parsed": _read(d / "parsed_episodes.json", {}), "decisions": _read(d / "editor_decisions.json", {})}


def save_decisions(run):
    (run["dir"] / "editor_decisions.json").write_text(json.dumps(run["decisions"], indent=2), encoding="utf-8")


def banner(m):
    if m.get("is_demo_or_offline"):
        st.warning("OFFLINE ENGINE: results are produced by deterministic rules, NOT by an LLM. "
                   "Treat them as a baseline shortlist for editor review.")
    else:
        st.info(f"LLM run: {m.get('llm_model')} (+ rule prefilter). Raw model replies are in raw_llm_responses/.")


# ------------------------------------------------------------------ pages
def page_dashboard(run, runs):
    m = run["manifest"]
    st.title("PromoForge")
    st.caption("AI-assisted promo discovery for microdrama - the editor makes the final call.")
    banner(m)
    st.subheader(m.get("show", "(no run loaded)"))
    c = st.columns(4)
    c[0].metric("Episodes processed", m.get("episodes", 0))
    c[1].metric("Scenes parsed", m.get("scenes", 0))
    c[2].metric("Candidates found", m.get("candidates", 0))
    c[3].metric("Seeds (raw moments)", m.get("seeds", 0))
    c = st.columns(4)
    c[0].metric("Accepted promos", m.get("accepted", 0))
    c[1].metric("Needs review", m.get("review", 0))
    c[2].metric("Rejected / dropped", m.get("rejected", 0))
    c[3].metric("Average accepted score", m.get("avg_accepted_score") or "-")
    st.caption(f"Status: {'completed' if m else 'no run yet'} | run time {m.get('run_seconds', '-')}s | "
               f"parse flags: {m.get('parse_flags', '-')} | LLM calls: {m.get('llm_calls', 0)}")
    if run["parsed"]:
        eps = pd.DataFrame([{"episode": e["episode_id"], "duration_s": e["duration"], "scenes": len(e["scenes"]),
                             "accepted": sum(1 for p in run["ranked"] if p["segments"][0]["episode"] == e["episode_id"])}
                            for e in run["parsed"]["episodes"]]).set_index("episode")
        st.bar_chart(eps["accepted"])
        st.caption("Accepted promos per episode (an episode with 0 simply had no strong moment).")
    st.divider()
    st.subheader("Process a show")
    up = st.file_uploader("Upload screenplay files (.txt) or one .zip", type=["txt", "md", "zip"], accept_multiple_files=True)
    engines = ["heuristic (offline, free)"] + (["llm (needs API key)"] if config.LLM_API_KEY else [])
    engine = st.selectbox("Engine", engines)
    if not config.LLM_API_KEY:
        st.caption("Set GEMINI_API_KEY in .env to enable the LLM engine.")
    if st.button("Start processing") and up:
        from src.parser import load_show
        from src.pipeline import run_pipeline
        name = Path(up[0].name).stem
        folder = UPLOAD_ROOT / name
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        for f in up:
            (folder / f.name).write_bytes(f.getvalue())
        bar = st.progress(0.0, text="Parsing and scoring...")
        show = load_show(folder)
        eng = "llm" if engine.startswith("llm") else "heuristic"
        client = None
        if eng == "llm":
            from src.llm_client import make_client
            client = make_client(raw_dir=OUT_ROOT / f"run_{name}" / "raw_llm_responses")
        run_pipeline(show, OUT_ROOT / f"run_{name}", engine=eng, client=client, log=lambda msg: None)
        bar.progress(1.0, text="Done - choose the new run in the sidebar.")
    elif st.button("Start processing (no file chosen)"):
        st.error("Upload at least one file first.")


def _card(p, run, key):
    s = p["segments"]
    st.markdown(f"**#{p.get('rank') or '-'}  |  {seg_text(p)}  |  {p['total_duration']:.1f}s  |  score {p['final_score']:.2f}**  "
                f"({p['decision']}, reliability {p['reliability']['level']})")
    st.write(f"Hook: {p['hook']}")
    st.write(f"Cliffhanger: {p['cliffhanger']}")
    st.caption(p["why_it_works"])
    sc = p["scores"]
    st.caption(" | ".join(f"{k.replace('_score', '')} {v:.1f}" for k, v in sc.items()))
    if p["weaknesses"] or p["context_check"].get("issues"):
        st.caption("Weaknesses: " + "; ".join(p["weaknesses"] + p["context_check"].get("issues", [])))
    with st.expander("Supporting dialogue / action"):
        for x in p["excerpt"]:
            st.text(f"{x['t']}  {(x['speaker'] or 'ACTION')[:18]:<18} {x['text'][:160]}")
    cols = st.columns(3)
    cur = run["decisions"].get(p["promo_id"], "undecided")
    if cols[0].button("Approve", key=f"a{key}"):
        run["decisions"][p["promo_id"]] = "approved"; save_decisions(run)
    if cols[1].button("Reject", key=f"r{key}"):
        run["decisions"][p["promo_id"]] = "rejected"; save_decisions(run)
    cols[2].caption(f"Editor decision: {run['decisions'].get(p['promo_id'], cur)}")


def page_ranked(run):
    st.title("Ranked promos")
    banner(run["manifest"])
    pool = run["ranked"] + run["review"]
    eps = sorted({p["segments"][0]["episode"] for p in pool})
    c = st.columns(4)
    sel = c[0].multiselect("Episode", eps)
    min_score = c[1].slider("Min score", 0.0, 10.0, 0.0, 0.1)
    sort = c[2].selectbox("Sort by", ["Rank (diversity-aware)", "Final score", "Episode", "Duration"])
    show_review = c[3].checkbox("Include 'needs review'", value=True)
    items = [p for p in pool if (show_review or p["decision"] == "accept") and p["final_score"] >= min_score
             and (not sel or p["segments"][0]["episode"] in sel)]
    keyf = {"Rank (diversity-aware)": lambda p: (p["rank"] or 10_000, -p["final_score"]),
            "Final score": lambda p: -p["final_score"], "Episode": lambda p: p["segments"][0]["episode"],
            "Duration": lambda p: p["total_duration"]}[sort]
    items.sort(key=keyf)
    st.caption(f"{len(items)} promos shown. Ranking applies a diversity penalty so near-duplicates sink.")
    for i, p in enumerate(items):
        with st.container():
            _card(p, run, f"{p['promo_id']}{i}")
            st.divider()


def page_details(run):
    st.title("Promo details")
    pool = run["ranked"] + run["review"]
    if not pool:
        st.info("No promos in this run."); return
    labels = {f"#{p.get('rank') or '-'}  Ep{p['segments'][0]['episode']}  {p['segments'][0]['start_ts']}  ({p['promo_id']})": p for p in pool}
    p = labels[st.selectbox("Promo", list(labels))]
    st.subheader(seg_text(p))
    n = p.get("narrative", {})
    st.markdown("**Hook-to-cliffhanger progression**")
    for k in ("hook", "development", "reveal", "open_question", "why_continue"):
        st.write(f"- {k.replace('_', ' ').title()}: {n.get(k, '')}")
    st.caption(n.get("method", ""))
    st.markdown("**Segments** (precision: " + p["validation"]["precision_note"] + ")")
    st.dataframe(pd.DataFrame([{k: s[k] for k in ("episode", "start_ts", "end_ts", "duration", "scene_ids", "source_lines")} for s in p["segments"]]))
    st.markdown("**Scores**")
    st.bar_chart(pd.Series(p["scores"]))
    cc = p["context_check"]
    st.markdown(f"**Standalone context check:** {cc.get('level', '?')} - depends on prior episodes: {cc.get('depends_on_prior_episodes')}")
    for i in cc.get("issues", []):
        st.write(f"- {i}")
    if cc.get("bridge_suggestion"):
        st.write(f"Context bridge: {cc['bridge_suggestion']}")
    st.markdown("**Evidence (verbatim from the screenplay)**")
    for e in p["evidence"]:
        st.write(f"- [{e['dimension']}] {e.get('timestamp', '')}: \"{e['quote']}\"")
    st.markdown("**Full selected dialogue and actions**")
    for x in p["excerpt"]:
        st.text(f"Ep{x['episode']} {x['t']}  {(x['speaker'] or 'ACTION')[:18]:<18} {x['text']}")
    st.markdown(f"**Reliability: {p['reliability']['level']}** - " + "; ".join(p["reliability"]["reasons"]))
    st.markdown("**Weaknesses:** " + ("; ".join(p["weaknesses"]) or "none flagged"))
    st.markdown("**Editor recommendation:** " + {"accept": "Shortlist - watch the footage, check the cut points.",
                                                  "review": "Borderline - " + "; ".join(p["decision_reasons"]),
                                                  "reject": "Not recommended."}[p["decision"]])


def page_quality(run):
    st.title("Quality report")
    rep, rej = run["report"], run["rejected"]
    st.subheader("Rejected / dropped candidates")
    if rej:
        df = pd.DataFrame([{"id": r["candidate_id"], "episode": r["episode"], "stage": r["stage"],
                            "score": r.get("final_score"), "reasons": "; ".join(r["reasons"])} for r in rej])
        st.dataframe(df)
        st.bar_chart(df["stage"].value_counts())
    st.subheader("Source / timestamp flags")
    st.dataframe(pd.DataFrame(rep.get("parse_flags", []) or [{"result": "no parse flags"}]))
    st.subheader("Validation")
    st.write(f"Failures: {len(rep.get('validation_failures', []))} | warnings: {len(rep.get('validation_warnings', []))}")
    st.write(f"Low-context candidates: {len(rep.get('low_context_candidates', []))} | bridged: {len(rep.get('bridged', []))} | "
             f"cross-episode: {len(rep.get('cross_episode', []))}")
    st.write(f"Model errors: {rep.get('model_errors') or 'none'}")
    st.subheader("Known limitations")
    for l in LIMITATIONS:
        st.write(f"- {l}")


def page_export(run):
    st.title("Export")
    ranked, review = run["ranked"], run["review"]
    approved_ids = {k for k, v in run["decisions"].items() if v == "approved"}
    approved = [p for p in ranked + review if p["promo_id"] in approved_ids]
    st.write(f"{len(ranked)} ranked | {len(review)} to review | {len(approved)} approved by the editor")
    csv = pd.DataFrame(promos_to_rows(ranked)).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Ranked promos (CSV)", csv, "ranked_promos.csv", "text/csv")
    if approved:
        st.download_button("Approved promos only (CSV)", pd.DataFrame(promos_to_rows(approved)).to_csv(index=False).encode("utf-8-sig"),
                           "approved_promos.csv", "text/csv")
    st.download_button("Complete results (JSON)", json.dumps({"manifest": run["manifest"], "ranked": ranked, "review": review}, indent=2),
                       "promoforge_results.json", "application/json")
    st.download_button("Rejected candidates (JSON)", json.dumps(run["rejected"], indent=2), "rejected_candidates.json", "application/json")
    best = {int(k): v for k, v in run["best"].items()}
    st.download_button("Editorial report (Markdown)", editorial_markdown(run["manifest"], ranked, review, best),
                       "editorial_report.md", "text/markdown")


PAGES = {"Dashboard": page_dashboard, "Ranked promos": page_ranked, "Promo details": page_details,
         "Quality report": page_quality, "Export": page_export}


def main():
    st.set_page_config(page_title="PromoForge", layout="wide")
    runs = list_runs()
    st.sidebar.title("PromoForge")
    page = st.sidebar.radio("Page", list(PAGES))
    if not runs:
        st.info("No run found yet. Run `python main.py run --input <folder>` or use Dashboard > Process a show.")
        run = {"dir": OUT_ROOT, "manifest": {}, "ranked": [], "review": [], "rejected": [], "report": {}, "best": {},
               "parsed": {}, "decisions": {}}
    else:
        run = load_run(st.sidebar.selectbox("Run", runs))
    fn = PAGES[page]
    fn(run, runs) if fn is page_dashboard else fn(run)


if __name__ == "__main__":
    main()
