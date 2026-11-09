"""Builds small synthetic screenplays in the Kuku format for tests (no confidential data used)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.timeutil import sec_to_ts


def make_episode_text(num=1, lines=None, scene_end=None, upper=False, header_num=None):
    lines = lines or [
        ("ANA", 0.0, 2.0, "You lied to me!"), ("BEN", 2.5, 4.0, "I had no choice."),
        ("ANA", 4.5, 7.0, "Who is she, Ben?"), ("BEN", 7.5, 9.0, "My wife."),
        ("ANA", 10.0, 12.0, "Your wife?!"), ("BEN", 13.0, 15.0, "I was going to tell you."),
        ("ANA", 16.0, 18.0, "Tell me what?"), ("BEN", 19.0, 21.0, "That I am the Duke."),
        ("ANA", 22.0, 24.0, "You are lying again."), ("BEN", 25.0, 27.0, "Open the door and see."),
        ("ANA", 28.0, 30.0, "Why is the king here?"),
    ]
    end = scene_end or max(l[2] for l in lines) + 1
    H = (lambda s: s.upper()) if upper else (lambda s: s)
    out = [f"EPISODE NUMBER: Episode {header_num or num}", "EPISODE LOGLINE: Test logline.",
           "CONTINUITY NOTES: Test.", "", "--- SCRIPT START ---", "", "SCENE 01",
           H("Scene heading: INT. HALL - DAY"), H(f"Scene start: 00:00.00"), H(f"Scene end: {sec_to_ts(end)}"), "",
           H("Actions and dialogue:"), ""]
    for who, s, e, text in lines:
        out += [f"**{who} (Tense):**", f"Start: ({sec_to_ts(s)}) | End: ({sec_to_ts(e)})", f'"{text}"', ""]
    out += ["--- SCRIPT END ---"]
    return "\n".join(out)
