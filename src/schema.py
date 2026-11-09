"""Plain dataclasses for the data model + validation of LLM JSON.
(We use stdlib dataclasses instead of pydantic so the core has zero dependencies.)"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Event:
    event_id: str
    episode: int
    scene_id: int
    kind: str                  # DIALOGUE | ACTION | CAMERA | SOUND_FX
    label: str                 # raw label e.g. 'LADY JULIANA (Shaken, Whispering)'
    speaker: Optional[str]
    tone: Optional[str]
    start: Optional[float]
    end: Optional[float]
    text: str
    line_start: int            # 1-based line numbers in the source file
    line_end: int


@dataclass
class Scene:
    scene_id: int
    heading: Optional[str]
    start: Optional[float]
    end: Optional[float]
    events: list = field(default_factory=list)


@dataclass
class Episode:
    episode_id: int
    source_file: str
    logline: str = ""
    continuity: str = ""
    title: Optional[str] = None
    scenes: list = field(default_factory=list)
    duration: float = 0.0
    flags: list = field(default_factory=list)

    def events(self):
        return [e for s in self.scenes for e in s.events]


@dataclass
class Show:
    name: str
    episodes: list
    flags: list = field(default_factory=list)


def to_dict(obj):
    return asdict(obj)


# ---------- LLM output validation ----------
SCORE_KEYS = ["hook_score", "cliffhanger_score", "standalone_clarity_score",
              "emotional_impact_score", "narrative_coherence_score",
              "promotional_potential_score"]


class SchemaError(ValueError):
    pass


def validate_evaluation(d: dict) -> dict:
    """Check an evaluation dict (from an LLM) has the required shape. Returns a cleaned copy."""
    if not isinstance(d, dict):
        raise SchemaError("evaluation is not an object")
    out = {}
    for k in SCORE_KEYS:
        if k not in d:
            raise SchemaError(f"missing {k}")
        v = d[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SchemaError(f"{k} is not a number")
        if not 0 <= v <= 10:
            raise SchemaError(f"{k}={v} outside 0-10")
        out[k] = float(v)
    ev = d.get("evidence")
    if not isinstance(ev, list) or not ev:
        raise SchemaError("evidence must be a non-empty list (scores need evidence)")
    for item in ev:
        if not isinstance(item, dict) or not item.get("quote") or "dimension" not in item:
            raise SchemaError("each evidence item needs 'dimension' and 'quote'")
    out["evidence"] = ev
    out["weaknesses"] = [str(w) for w in d.get("weaknesses", [])]
    dec = str(d.get("decision", "review")).lower()
    if dec not in ("accept", "reject", "review"):
        raise SchemaError(f"bad decision {dec!r}")
    out["decision"] = dec
    for k in ("hook", "cliffhanger", "why_it_works"):
        if k in d:
            out[k] = str(d[k])
    if isinstance(d.get("narrative"), dict):
        out["narrative"] = d["narrative"]
    return out


def validate_candidates(d) -> list:
    """Validate the LLM candidate-discovery response: {'candidates':[{start,end,...}]}"""
    if isinstance(d, dict):
        d = d.get("candidates")
    if not isinstance(d, list):
        raise SchemaError("expected a list of candidates")
    out = []
    for c in d:
        if not isinstance(c, dict) or "start" not in c or "end" not in c:
            raise SchemaError("candidate needs start and end")
        out.append(c)
    return out
