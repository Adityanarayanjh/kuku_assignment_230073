import json
import re
import tempfile
import unittest
from pathlib import Path
from helpers import make_episode_text
from src.parser import load_show
from src.pipeline import run_pipeline
from src.validator import validate_promo
from src.llm_client import LLMClient
import config


def synthetic_show():
    d = Path(tempfile.mkdtemp())
    long_lines = []
    t = 0.0
    texts = ["You lied to me!", "I had no choice.", "Who is she?", "My wife.", "Your wife?!", "I will destroy you!",
             "Tell me the truth.", "I am the Duke.", "You are lying.", "Open the door.", "Why is the king here?",
             "Guards, arrest him!", "He knew the secret.", "Run!", "What did you do?", "It is too late."]
    for i, tx in enumerate(texts * 2):
        long_lines.append(("ANA" if i % 2 == 0 else "BEN", t, t + 2.0, tx)); t += 2.6
    for n in (1, 2):
        (d / f"episode-00{n}-x.txt").write_text(make_episode_text(n, lines=long_lines), encoding="utf-8")
    return load_show(d)


class ScriptedLLM(LLMClient):
    """Behaves like a (well-behaved) model: reads the prompt and answers with quotes that exist."""
    provider = "scripted"

    def __init__(self, **kw):
        super().__init__("scripted", sleep=lambda s: None, **kw)

    def _call(self, prompt, system=None):
        if "SCREENPLAY:" in prompt:
            return json.dumps({"candidates": [{"start": "00:05.20", "end": "00:40.00", "hook_quote": "x", "why": "test"}]})
        quote = re.findall(r"\d\d:\d\d\.\d\d \w+: (.+)", prompt)[0]
        return json.dumps({**{k: 7.5 for k in config.WEIGHTS}, "evidence": [{"dimension": "hook", "quote": quote}],
                           "weaknesses": [], "hook": "h", "cliffhanger": "c", "why_it_works": "w", "decision": "accept"})


class TestPipeline(unittest.TestCase):
    def test_heuristic_end_to_end_invariants(self):
        show = synthetic_show()
        out = tempfile.mkdtemp()
        res = run_pipeline(show, out, engine="heuristic", log=lambda m: None)
        by_id = {e.episode_id: e for e in show.episodes}
        for name in ("ranked_promos.json", "rejected_candidates.json", "candidates.json", "evaluation_report.json",
                     "parsed_episodes.json", "ranked_promos.csv", "run_manifest.json", "editorial_report.md"):
            self.assertTrue((Path(out) / name).exists(), name)
        self.assertTrue(res["manifest"]["is_demo_or_offline"])
        for p in res["ranked"]:
            v = validate_promo(p, by_id)                                     # independent re-validation
            self.assertTrue(v["passed"], v["failures"])
            self.assertTrue(config.MIN_DURATION - 0.01 <= p["total_duration"] <= config.MAX_DURATION + 0.01)
            self.assertEqual(p["decision"], "accept")
            self.assertIn("heuristic", p["engine"])

    def test_llm_path_with_scripted_model(self):
        show = synthetic_show()
        out = tempfile.mkdtemp()
        client = ScriptedLLM(raw_dir=Path(out) / "raw")
        res = run_pipeline(show, out, engine="llm", client=client, llm_eval_top=2, log=lambda m: None)
        self.assertGreater(client.calls, 0)
        cand = json.loads((Path(out) / "candidates.json").read_text())
        self.assertTrue(any(c["seed_source"] == "llm" for c in cand["candidates"]))   # LLM windows reach evaluation
        self.assertTrue(list((Path(out) / "raw").glob("*.txt")))             # raw replies kept
        for p in res["promos"]:
            self.assertIn("llm:scripted", p["engine"])

    def test_hallucinated_evidence_is_caught(self):
        class Liar(ScriptedLLM):
            def _call(self, prompt, system=None):
                if "SCREENPLAY:" in prompt:
                    return json.dumps({"candidates": []})
                return json.dumps({**{k: 9 for k in config.WEIGHTS}, "evidence": [{"dimension": "hook", "quote": "invented line"}],
                                   "decision": "accept"})
        show = synthetic_show(); out = tempfile.mkdtemp()
        res = run_pipeline(show, out, engine="llm", client=Liar(), llm_eval_top=2, log=lambda m: None)
        self.assertEqual(len(res["ranked"]), 0)                              # nothing with fake evidence is accepted
        self.assertTrue(res["rejected"])


if __name__ == "__main__":
    unittest.main()
