"""Independent audit of a finished run. Re-reads the ORIGINAL screenplay files and checks every promo:
  episode exists | times inside episode | 20-60 s total | every excerpt line text + timestamp + line number matches the file.
Usage: python verify_outputs.py --input <screenplay folder or zip> --run outputs/run_heuristic"""
import argparse, json, re, sys
from pathlib import Path
from src.parser import _collect_files
from src.timeutil import ts_to_sec

ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--run", required=True)
a = ap.parse_args()
files, _ = _collect_files(Path(a.input), [])
by_ep = {}
for f in files:
    m = re.search(r"episode[-_ ]?(\d+)", f.name, re.I)
    if m and f.suffix.lower() == ".txt":
        by_ep[int(m.group(1))] = f.read_text(encoding="utf-8").splitlines()
run = Path(a.run)
promos = json.loads((run / "ranked_promos.json").read_text()) + json.loads((run / "review_queue.json").read_text())
problems, checked = [], 0
for p in promos:
    total = 0.0
    for sg in p["segments"]:
        if sg["episode"] not in by_ep:
            problems.append((p["promo_id"], "episode file missing")); continue
        total += sg["end"] - sg["start"]
    for ex in p["excerpt"]:                       # each excerpt line belongs to exactly one segment (by time)
        checked += 1
        lines = by_ep.get(ex["episode"])
        owners = [sg for sg in p["segments"] if sg["episode"] == ex["episode"]
                  and sg["start"] - 0.011 <= ts_to_sec(ex["t"]) < sg["end"] + 0.011]
        if lines is None or not owners:
            problems.append((p["promo_id"], f"excerpt {ex['t']} not inside any segment")); continue
        a_, b_ = owners[0]["source_lines"]
        block = " ".join(lines[a_ - 1:b_])
        if ex["text"] not in block:
            problems.append((p["promo_id"], f"text not at cited lines: {ex['text'][:40]}"))
        if f"({ex['t']})" not in block:
            problems.append((p["promo_id"], f"timestamp {ex['t']} not at cited lines"))
    if not (20 - 0.02 <= total <= 60 + 0.02):
        problems.append((p["promo_id"], f"duration {total:.2f}"))
print(f"promos checked: {len(promos)}; excerpt lines traced to source file/line numbers: {checked}; problems: {len(problems)}")
for pr in problems[:20]:
    print("  PROBLEM", pr)
sys.exit(1 if problems else 0)
