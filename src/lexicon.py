"""Cheap, deterministic signals for melodrama microdrama. Genre-generic (not tuned to one show).
Used ONLY for the offline engine and for pre-filtering; LLM mode judges context itself."""
import re

def _rx(words):
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I)

# category -> (regex, weight per category hit). Each category counts once per event.
CATEGORIES = {
    "betrayal":   (_rx(["betray\\w*", "traitor", "lied to me", "you lied", "liar", "cheat\\w*", "deceiv\\w*", "backstab\\w*",
                        "abandon\\w*", "swore you", "you knew"]), 2.0),
    "secret":     (_rx(["secret", "truth", "hidden", "hiding", "confess\\w*", "reveal\\w*", "who are you", "real name",
                        "identity", "you are not", "never told", "discover\\w*", "proof", "evidence"]), 2.0),
    "threat":     (_rx(["kill", "die", "dead", "death", "murder\\w*", "poison\\w*", "arsenic", "destroy", "ruin",
                        "you will pay", "you'll pay", "regret", "warn\\w*", "threat\\w*", "hang", "execut\\w*",
                        "arrest\\w*", "prison", "dungeon", "last chance", "or else"]), 2.0),
    "violence":   (_rx(["slap\\w*", "strike\\w*", "struck", "shove\\w*", "dagger", "sword", "blade", "knife", "gun",
                        "pistol", "blood\\w*", "bleed\\w*", "stab\\w*", "explod\\w*", "explosion", "fire", "flames",
                        "burst\\w*", "slam\\w*", "collaps\\w*", "faint\\w*", "drag\\w*", "grab\\w*", "seiz\\w*"]), 2.0),
    "romance":    (_rx(["kiss\\w*", "marry me", "my wife", "my husband", "i love you", "love you", "beloved",
                        "bride", "wedding", "wed\\w*", "embrace\\w*", "heartbeat", "mine"]), 1.5),
    "status":     (_rx(["duke", "duchess", "prince", "princess", "king", "queen", "heir", "crown", "throne",
                        "royal\\w*", "lord", "estate", "inherit\\w*", "title"]), 1.0),
    "shock":      (_rx(["impossible", "how could", "how dare", "what did you", "what do you mean", "you can't",
                        "no\\.\\.\\.", "gasp\\w*", "stun\\w*", "shock\\w*", "horrif\\w*", "alive", "pregnan\\w*",
                        "divorce", "annul\\w*"]), 1.5),
    "ultimatum":  (_rx(["get out", "leave", "never forgive", "i forbid", "choose", "decide", "tonight", "by dawn",
                        "one more", "enough", "no more", "you have until", "if you", "swear"]), 1.0),
}

TONE_INTENSE = _rx(["furious", "enraged", "shocked", "stunned", "horrified", "menac\\w*", "lethal\\w*", "chilling",
                    "venom\\w*", "desperate", "sobbing", "screaming", "shouting", "cold", "merciless", "ominous",
                    "threatening", "trembling", "panicked", "seething", "gloating", "mocking", "cruel", "scornful"])

CONNECTIVE_START = re.compile(r"^(and|but|so|because|yes|no|then|well|not|also|or|nor|that|those|it|he|she|they|him|her"
                              r"|his|them|this|these|which|who|exactly|indeed|still|just|even|now|too)\b", re.I)
PRONOUN_START = re.compile(r"^(he|she|they|him|her|his|them|it|that|this)\b", re.I)
RESOLVING = re.compile(r"\b(thank you|thanks|goodbye|farewell|forgive me|i understand|very well|as you wish|"
                       r"of course|i promise|all is well|it is done|rest now|peace)\b", re.I)
RELATION = re.compile(r"\b(husband|wife|mother|father|mama|sister|brother|daughter|son|fianc\w+|bride|groom|"
                      r"duke|duchess|lord|lady|master|maid|servant|enemy|stranger|soldier|queen|king)\b", re.I)
PROPER_NOUN = re.compile(r"(?<![.!?\"“]\s)(?<!^)\b([A-Z][a-z]{2,})\b")


# Calibration (added after the first full run): raw line scores had p99 = 4.6 and max = 6.8 on this show,
# which compressed every downstream 0-10 score. x2 puts p99 near 9. One fixed constant, not tuned per clip.
INTENSITY_SCALE = 2.0


def intensity(kind: str, text: str, tone: str = None):
    """Return (score 0-10, [categories hit]) for one dialogue/action event."""
    if kind not in ("DIALOGUE", "ACTION"):
        return 0.0, []
    score, hits = 0.0, []
    for name, (rx, w) in CATEGORIES.items():
        if rx.search(text):
            score += w
            hits.append(name)
    q = min(text.count("?"), 2) * 0.6
    ex = min(text.count("!"), 2) * 0.6
    score += q + ex
    if text.rstrip().endswith("..."):
        score += 0.5
    if tone and TONE_INTENSE.search(tone):
        score += 1.2
    words = len(text.split())
    if kind == "DIALOGUE" and words <= 7 and score > 0:
        score += 0.8                      # short punchy line
    if kind == "ACTION" and words > 30:
        score *= 0.8                      # long descriptive action is less punchy
    return min(score * INTENSITY_SCALE, 10.0), hits
