"""Stage A: screenplay ingestion.

Reads the Kuku screenplay text format:
    EPISODE NUMBER / EPISODE LOGLINE / CONTINUITY NOTES
    --- SCRIPT START ---
    SCENE 01 / Scene heading / Scene start / Scene end
    **LABEL (tone):**   <- event header (CAMERA, ACTION, SOUND FX or a speaker)
    Start: (mm:ss.cc) | End: (mm:ss.cc)
    text...
Header casing differs between files (episode 6 is upper-case) so matching is case-insensitive.
Nothing is invented: if a timestamp is missing or malformed we store None and add a flag.
"""
import re
import tempfile
import zipfile
from pathlib import Path

from .schema import Event, Scene, Episode, Show
from .timeutil import ts_to_sec

RE_EP_NUM = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(\d+)", re.I)
RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.*)", re.I)
RE_CONT = re.compile(r"^CONTINUITY NOTES:\s*(.*)", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_HEADING = re.compile(r"^SCENE HEADING:\s*(.*)", re.I)
RE_SSTART = re.compile(r"^SCENE START:\s*(\S+)", re.I)
RE_SEND = re.compile(r"^SCENE END:\s*(\S+)", re.I)
RE_BLOCK = re.compile(r"^\*\*(.+):\*\*\s*$")
RE_TS = re.compile(r"Start:\s*\(([^)]*)\)\s*(?:\|\s*End:\s*\(([^)]*)\))?", re.I)
RE_PAREN = re.compile(r"\(([^)]*)\)")
MODS = {"O.S.", "V.O.", "O.C.", "CONT'D", "CONT’D"}
NON_SPEECH = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SOUND_FX", "SFX": "SOUND_FX"}


def _flag(ep_flags, ep, line, typ, detail):
    ep_flags.append({"episode": ep, "line": line, "type": typ, "detail": detail})


def _safe_ts(raw, flags, ep, line, what):
    if raw is None:
        return None
    try:
        return ts_to_sec(raw)
    except ValueError:
        _flag(flags, ep, line, "malformed_timestamp", f"{what}: {raw!r}")
        return None


def _split_label(label):
    """'LADY JULIANA (Shaken, Whispering)' -> ('LADY JULIANA', 'Shaken, Whispering')"""
    name = re.split(r"\s*\(", label, maxsplit=1)[0].strip()
    parens = RE_PAREN.findall(label)
    tones = [p for p in parens if p.strip().upper() not in MODS]
    return name, (tones[-1].strip() if tones else None)


def parse_episode_text(text: str, source_file: str = "") -> Episode:
    lines = text.splitlines()
    fn_num = re.search(r"episode[-_ ]?(\d+)", Path(source_file).name, re.I)
    ep = Episode(episode_id=int(fn_num.group(1)) if fn_num else -1, source_file=Path(source_file).name)
    flags = ep.flags
    header_num = None
    scene = None
    in_script = False
    counter = 0
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        n = i + 1
        if not in_script:
            m = RE_EP_NUM.match(ln)
            if m:
                header_num = int(m.group(1))
            m = RE_LOGLINE.match(ln)
            if m:
                ep.logline = m.group(1).strip()
            m = RE_CONT.match(ln)
            if m:
                ep.continuity = m.group(1).strip()
            if ln.upper().startswith("--- SCRIPT START"):
                in_script = True
            i += 1
            continue
        if ln.upper().startswith("--- SCRIPT END"):
            break
        m = RE_SCENE.match(ln)
        if m:
            scene = Scene(scene_id=int(m.group(1)), heading=None, start=None, end=None)
            ep.scenes.append(scene)
            i += 1
            continue
        if scene is not None:
            m = RE_HEADING.match(ln)
            if m:
                scene.heading = m.group(1).strip(); i += 1; continue
            m = RE_SSTART.match(ln)
            if m:
                scene.start = _safe_ts(m.group(1), flags, ep.episode_id, n, "scene start"); i += 1; continue
            m = RE_SEND.match(ln)
            if m:
                scene.end = _safe_ts(m.group(1), flags, ep.episode_id, n, "scene end"); i += 1; continue
        m = RE_BLOCK.match(ln)
        if m:
            if scene is None:   # content before any SCENE line
                scene = Scene(scene_id=0, heading=None, start=None, end=None)
                ep.scenes.append(scene)
                _flag(flags, ep.episode_id, n, "event_outside_scene", "event before first SCENE header")
            label = m.group(1).strip()
            start = end = None
            j = i + 1
            ts_line_no = None
            if j < len(lines):
                mt = RE_TS.search(lines[j])
                if mt:
                    start = _safe_ts(mt.group(1), flags, ep.episode_id, j + 1, "event start")
                    end = _safe_ts(mt.group(2), flags, ep.episode_id, j + 1, "event end")
                    ts_line_no = j + 1
                    j += 1
            if ts_line_no is None:
                _flag(flags, ep.episode_id, n, "missing_timestamp", f"event {label!r} has no Start/End line")
            elif end is None and start is not None:
                _flag(flags, ep.episode_id, ts_line_no, "missing_end", f"event {label!r} has no end time")
            body = []
            while j < len(lines):
                s = lines[j].strip()
                if (not s) or RE_BLOCK.match(s) or RE_SCENE.match(s) or s.upper().startswith("--- SCRIPT END"):
                    break
                body.append(s)
                j += 1
            name, tone = _split_label(label)
            key = name.upper()
            kind = NON_SPEECH.get(key, "DIALOGUE")
            txt = " ".join(body).strip()
            if kind == "DIALOGUE":
                txt = txt.strip('"“”').strip()
            counter += 1
            scene.events.append(Event(
                event_id=f"E{ep.episode_id:03d}-S{scene.scene_id:02d}-{counter:03d}",
                episode=ep.episode_id, scene_id=scene.scene_id, kind=kind, label=label,
                speaker=name if kind == "DIALOGUE" else None, tone=tone,
                start=start, end=end, text=txt, line_start=n, line_end=max(j, n)))
            i = j
            continue
        i += 1

    if header_num is None:
        _flag(flags, ep.episode_id, 0, "missing_episode_header", "no 'EPISODE NUMBER' line; using filename")
    elif ep.episode_id == -1:
        ep.episode_id = header_num
    elif header_num != ep.episode_id:
        _flag(flags, ep.episode_id, 0, "episode_number_mismatch",
              f"file name says {ep.episode_id}, header says {header_num}; using header")
        ep.episode_id = header_num
    if not in_script:
        _flag(flags, ep.episode_id, 0, "no_script_start", "'--- SCRIPT START ---' not found")
    _post_checks(ep)
    return ep


def _post_checks(ep: Episode):
    """Sanity checks on timestamps; never repair, only flag."""
    flags, eid = ep.flags, ep.episode_id
    max_end = 0.0
    prev_end = None
    for sc in ep.scenes:
        if sc.start is None or sc.end is None:
            _flag(flags, eid, 0, "scene_missing_timestamp", f"scene {sc.scene_id} lacks start/end")
        else:
            if sc.end <= sc.start:
                _flag(flags, eid, 0, "scene_bad_order", f"scene {sc.scene_id} end <= start")
            if prev_end is not None and abs(sc.start - prev_end) > 0.05:
                _flag(flags, eid, 0, "scene_gap_or_overlap",
                      f"scene {sc.scene_id} starts {sc.start} but previous ended {prev_end}")
            prev_end = sc.end
            max_end = max(max_end, sc.end)
        for ev in sc.events:
            if ev.start is not None and ev.end is not None:
                if ev.end < ev.start:
                    _flag(flags, eid, ev.line_start, "event_bad_order", ev.event_id)
                max_end = max(max_end, ev.end)
                if sc.start is not None and sc.end is not None and \
                        (ev.start < sc.start - 0.05 or ev.end > sc.end + 0.05):
                    _flag(flags, eid, ev.line_start, "event_outside_scene_range", ev.event_id)
            if ev.kind == "DIALOGUE" and not ev.text:
                _flag(flags, eid, ev.line_start, "empty_dialogue", ev.event_id)
    ep.duration = round(max_end, 3)
    if not ep.events():
        _flag(flags, eid, 0, "no_events", "episode has no parsed events")


def _collect_files(path: Path, tmp_holder: list):
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.mkdtemp(prefix="promoforge_")
        tmp_holder.append(tmp)
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = Path(tmp)
    if path.is_file():
        return [path], path.stem
    files = sorted(p for p in path.rglob("*") if p.is_file())
    name = path.name
    # if the zip contained a single top-level folder, use its name
    subs = [p for p in path.iterdir() if p.is_dir()]
    if len(subs) == 1 and not [p for p in path.iterdir() if p.is_file()]:
        name = subs[0].name
    return files, name


def load_show(path) -> Show:
    """Load every supported screenplay file under a folder / .zip / single file."""
    path = Path(path)
    holder = []
    files, name = _collect_files(path, holder)
    episodes, show_flags = [], []
    for f in files:
        if f.suffix.lower() not in (".txt", ".md"):
            show_flags.append({"episode": None, "line": 0, "type": "unsupported_file", "detail": f.name})
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        episodes.append(parse_episode_text(text, str(f)))
    episodes.sort(key=lambda e: e.episode_id)
    seen = {}
    for e in episodes:
        if e.episode_id in seen:
            show_flags.append({"episode": e.episode_id, "line": 0, "type": "duplicate_episode",
                               "detail": f"{e.source_file} duplicates {seen[e.episode_id]}"})
        seen[e.episode_id] = e.source_file
    ids = sorted(seen)
    if ids:
        missing = [i for i in range(ids[0], ids[-1] + 1) if i not in seen]
        if missing:
            show_flags.append({"episode": None, "line": 0, "type": "missing_episodes",
                               "detail": f"gaps in numbering: {missing}"})
    return Show(name=name, episodes=episodes, flags=show_flags)
