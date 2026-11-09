"""Central configuration. Everything is overridable with environment variables / .env."""
import os
from pathlib import Path

ROOT = Path(__file__).parent


def _load_dotenv():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY", "")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", str(ROOT / "cache")))

# ---- promo rules from the assignment ----
MIN_DURATION = 20.0
MAX_DURATION = 60.0
HOOK_WINDOW = 3.0          # hook must land in the first 3 s

# ---- ranking ----
WEIGHTS = {
    "hook_score": 0.20,
    "cliffhanger_score": 0.20,
    "standalone_clarity_score": 0.20,
    "emotional_impact_score": 0.15,
    "narrative_coherence_score": 0.15,
    "promotional_potential_score": 0.10,
}
# hard gates (a promo cannot be accepted by being great elsewhere). Policy: "weak" = below 6/10,
# "accept" = overall >= 7/10. After the intensity calibration the first thresholds (5 / 6) accepted 72 of 195
# candidates and 44/50 episodes, i.e. not selective, so they were raised once.
HOOK_MIN = 6.0
CLIFF_MIN = 6.0
CLARITY_REVIEW_BELOW = 5.0
ACCEPT_MIN_FINAL = 7.0
GATE_CAP = 7.5             # if min(hook, cliff) < GATE_SOFT, final score capped
GATE_SOFT = 7.0
DIVERSITY_LAMBDA = 1.5     # penalty strength for similarity to already-selected promos
PER_EPISODE_CANDIDATES = 6
