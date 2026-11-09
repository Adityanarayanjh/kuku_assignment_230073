"""Stage F (part 2): diversity-aware ranking (greedy, MMR-style).
Pick the best promo, then repeatedly pick the one with the best (score - lambda * similarity-to-already-picked)."""
import config

DROP = {"lady", "lord", "the", "mysterious", "old", "young", "elder", "noble", "blond", "bearded", "head", "dowager",
        "high", "crown", "royal", "scarred", "bald", "tall", "third", "thug", "off-screen"}


def _chars(p):
    out = set()
    for c in p.get("characters", []):
        toks = [t for t in c.lower().replace("'s", "").split() if t not in DROP]
        out.add(toks[-1] if toks else c.lower())
    return out


def similarity(a, b) -> float:
    ea = {s["episode"] for s in a["segments"]}
    eb = {s["episode"] for s in b["segments"]}
    same_ep = 1.0 if ea & eb else (0.3 if any(abs(x - y) <= 1 for x in ea for y in eb) else 0.0)
    ca, cb = _chars(a), _chars(b)
    jac = len(ca & cb) / len(ca | cb) if ca | cb else 0.0
    same_type = 1.0 if a.get("hook_type") == b.get("hook_type") else 0.0
    return round(0.4 * same_ep + 0.35 * jac + 0.25 * same_type, 3)


def rank_diverse(promos, lam=config.DIVERSITY_LAMBDA):
    pool = sorted(promos, key=lambda p: -p["final_score"])
    chosen = []
    while pool:
        best, best_adj, best_sim, best_with = None, -1e9, 0.0, None
        for p in pool:
            sims = [(similarity(p, c), c["promo_id"]) for c in chosen]
            sim, with_ = max(sims, default=(0.0, None))
            adj = p["final_score"] - lam * sim
            if adj > best_adj:
                best, best_adj, best_sim, best_with = p, adj, sim, with_
        pool.remove(best)
        best["diversity"] = {"penalty": round(config.DIVERSITY_LAMBDA * best_sim, 3), "max_similarity": best_sim,
                             "most_similar_to": best_with, "adjusted_score": round(best_adj, 3)}
        chosen.append(best)
        best["rank"] = len(chosen)
    return chosen
