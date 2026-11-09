"""Stage F (part 1): decision rules. Scores come from an evaluator; the rules live here in plain Python."""
import config


def decide(promo: dict, validation: dict) -> dict:
    sc, reasons = promo["scores"], []
    reject = []
    if not validation["passed"]:
        reject.append("failed validation: " + "; ".join(validation["failures"][:2]))
    if sc["hook_score"] < config.HOOK_MIN:
        reject.append(f"weak hook ({sc['hook_score']:.1f} < {config.HOOK_MIN})")
    if sc["cliffhanger_score"] < config.CLIFF_MIN:
        reject.append(f"no real cliffhanger ({sc['cliffhanger_score']:.1f} < {config.CLIFF_MIN})")
    if promo["final_score"] < config.ACCEPT_MIN_FINAL - 1.0:
        reject.append(f"final score too low ({promo['final_score']:.2f})")
    if reject:
        return {"decision": "reject", "reasons": reject}
    review = []
    if sc["standalone_clarity_score"] < config.CLARITY_REVIEW_BELOW:
        review.append(f"low standalone clarity ({sc['standalone_clarity_score']:.1f})")
    if promo["final_score"] < config.ACCEPT_MIN_FINAL:
        review.append(f"final score below accept threshold ({promo['final_score']:.2f} < {config.ACCEPT_MIN_FINAL})")
    if len({s['episode'] for s in promo['segments']}) > 1:
        review.append("cross-episode combination needs coherence review")
    elif len(promo["segments"]) > 1:
        review.append("multi-segment cut needs editor check")
    if validation["warnings"]:
        review += [w for w in validation["warnings"] if not w.startswith("cross_episode")]
    if review:
        return {"decision": "review", "reasons": review}
    return {"decision": "accept", "reasons": ["passed all gates"]}
