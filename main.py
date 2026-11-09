"""PromoForge command line.
  python main.py run --input "data/input/<show folder or .zip>" --engine heuristic
  python main.py run --input ... --engine llm            (needs GEMINI_API_KEY in .env)
  python main.py estimate --input ...                    (token / call estimate for an LLM run)"""
import argparse
import sys
from pathlib import Path

import config
from src.parser import load_show
from src.pipeline import run_pipeline
from src.candidate_extractor import transcript_for_llm


def parse_eps(s):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def cmd_estimate(show, per_ep_eval=3):
    chars = sum(len(transcript_for_llm(e)) for e in show.episodes)
    n = len(show.episodes)
    disc_tokens = chars / 4                       # ~4 characters per token (rough rule of thumb)
    eval_calls = n * per_ep_eval
    eval_tokens = eval_calls * 1500               # ~1.5k tokens per excerpt+prompt (assumption)
    print(f"episodes: {n}; screenplay text sent for discovery: {chars:,} chars ~ {disc_tokens:,.0f} input tokens")
    print(f"discovery calls: {n} (1 per episode); evaluation calls: ~{eval_calls} (top {per_ep_eval}/episode)")
    print(f"evaluation input tokens ~ {eval_tokens:,.0f}; output tokens ~ {(n + eval_calls) * 600:,.0f} (assumed 600/call)")
    print("Cost = tokens x your model's current per-token price (not hard-coded here; check the provider's price page).")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promoforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--input", required=True)
    r.add_argument("--engine", choices=["heuristic", "llm"], default="heuristic")
    r.add_argument("--out", default=None); r.add_argument("--episodes", default=None, help="e.g. 1-5,9")
    r.add_argument("--llm-eval-top", type=int, default=3)
    e = sub.add_parser("estimate"); e.add_argument("--input", required=True)
    a = ap.parse_args(argv)
    show = load_show(a.input)
    if a.cmd == "estimate":
        return cmd_estimate(show)
    out = a.out or str(config.ROOT / "outputs" / f"run_{a.engine}")
    client = None
    if a.engine == "llm":
        from src.llm_client import make_client
        client = make_client(raw_dir=Path(out) / "raw_llm_responses")
    res = run_pipeline(show, out, engine=a.engine, client=client, episode_filter=parse_eps(a.episodes),
                       llm_eval_top=a.llm_eval_top)
    m = res["manifest"]
    print(f"\nDone: {m['accepted']} accepted, {m['review']} review, {m['rejected']} rejected. Outputs in {out}")


if __name__ == "__main__":
    main()
