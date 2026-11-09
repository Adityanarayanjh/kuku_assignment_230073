import copy
import unittest
from helpers import make_episode_text
from src.parser import parse_episode_text
from src.validator import validate_promo
from src.ranker import decide
from src.schema import validate_evaluation, SchemaError
from src.boundary_optimizer import nms
from src.window import Timeline
from src.evaluator import final_score
from src.diversity import rank_diverse
import config

EP = parse_episode_text(make_episode_text(1), "episode-001.txt")
EPS = {1: EP}


def good_promo():
    return {"promo_id": "T1", "engine": "test", "total_duration": 24.0,
            "segments": [{"episode": 1, "start": 0.0, "end": 24.0, "duration": 24.0}],
            "evidence": [{"dimension": "hook", "quote": "You lied to me!"}]}


class TestValidator(unittest.TestCase):
    def test_good(self):
        v = validate_promo(good_promo(), EPS)
        self.assertTrue(v["passed"], v["failures"])

    def test_missing_episode(self):
        p = good_promo(); p["segments"][0]["episode"] = 99
        self.assertFalse(validate_promo(p, EPS)["passed"])

    def test_start_after_end(self):
        p = good_promo(); p["segments"][0].update(start=20.0, end=10.0, duration=-10)
        self.assertFalse(validate_promo(p, EPS)["passed"])

    def test_too_short_and_too_long(self):
        p = good_promo(); p["segments"][0].update(end=10.0, duration=10.0); p["total_duration"] = 10.0
        v = validate_promo(p, EPS); self.assertFalse(v["passed"])
        self.assertTrue(any("min_duration" in f for f in v["failures"]))
        p = good_promo(); p["segments"][0].update(end=70.0, duration=70.0); p["total_duration"] = 70.0
        self.assertTrue(any("max_duration" in f or "inside_episode" in f for f in validate_promo(p, EPS)["failures"]))

    def test_exact_boundaries_allowed(self):
        p = good_promo(); p["segments"][0].update(start=0.0, end=20.0, duration=20.0); p["total_duration"] = 20.0
        self.assertTrue(validate_promo(p, EPS)["passed"])

    def test_total_duration_mismatch(self):
        p = good_promo(); p["total_duration"] = 30.0
        self.assertFalse(validate_promo(p, EPS)["passed"])

    def test_multi_segment_sum_and_order(self):
        p = {"promo_id": "T2", "engine": "test", "total_duration": 28.0, "evidence": [],
             "segments": [{"episode": 1, "start": 0.0, "end": 8.0, "duration": 8.0},
                          {"episode": 1, "start": 10.0, "end": 30.0, "duration": 20.0}]}
        self.assertTrue(validate_promo(p, EPS)["passed"])
        p["segments"].reverse()
        self.assertFalse(validate_promo(p, EPS)["passed"])      # out of order

    def test_no_source_content(self):
        p = good_promo(); p["segments"][0].update(start=31.0, end=31.0 + 24, duration=24.0)
        self.assertFalse(validate_promo(p, EPS)["passed"])

    def test_evidence_must_exist_in_source(self):
        p = good_promo(); p["evidence"].append({"dimension": "hook", "quote": "I am your father"})
        v = validate_promo(p, EPS)
        self.assertFalse(v["passed"]); self.assertTrue(any("evidence_traceable" in f for f in v["failures"]))

    def test_cut_through_line_is_warning(self):
        p = good_promo(); p["segments"][0].update(end=25.5, duration=25.5); p["total_duration"] = 25.5
        v = validate_promo(p, EPS)
        self.assertTrue(any("not_mid_line" in w for w in v["warnings"]))


class TestRulesAndSchema(unittest.TestCase):
    def _promo(self, **over):
        sc = {k: 8.0 for k in config.WEIGHTS}; sc.update(over)
        return {"scores": sc, "final_score": final_score(sc), "segments": [{"episode": 1}]}

    def test_weak_hook_never_accepted(self):
        p = self._promo(hook_score=3.0)                       # everything else is 8
        self.assertEqual(decide(p, {"passed": True, "warnings": [], "failures": []})["decision"], "reject")

    def test_no_cliffhanger_never_accepted(self):
        p = self._promo(cliffhanger_score=2.0)
        self.assertEqual(decide(p, {"passed": True, "warnings": [], "failures": []})["decision"], "reject")

    def test_failed_validation_rejects(self):
        p = self._promo()
        self.assertEqual(decide(p, {"passed": False, "warnings": [], "failures": ["x"]})["decision"], "reject")

    def test_low_clarity_goes_to_review(self):
        p = self._promo(standalone_clarity_score=3.0)
        self.assertEqual(decide(p, {"passed": True, "warnings": [], "failures": []})["decision"], "review")

    def test_gate_caps_final_score(self):
        sc = {k: 10.0 for k in config.WEIGHTS}; sc["hook_score"] = 2.0
        self.assertLessEqual(final_score(sc), config.GATE_CAP)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(config.WEIGHTS.values()), 1.0)

    def test_invalid_llm_evaluation_rejected(self):
        good = {k: 7 for k in config.WEIGHTS}; good["evidence"] = [{"dimension": "hook", "quote": "x"}]
        validate_evaluation(good)
        for mutate in (lambda d: d.pop("hook_score"), lambda d: d.update(hook_score=11),
                       lambda d: d.update(hook_score="high"), lambda d: d.update(evidence=[]),
                       lambda d: d.update(decision="maybe"), lambda d: d.update(hook_score=True)):
            d = copy.deepcopy(good); mutate(d)
            with self.assertRaises(SchemaError):
                validate_evaluation(d)
        with self.assertRaises(SchemaError):
            validate_evaluation("not a dict")


class TestDedupeAndDiversity(unittest.TestCase):
    def test_nms_drops_overlap(self):
        c = [{"candidate_id": "a", "window": (0, 30), "final_score": 8}, {"candidate_id": "b", "window": (5, 32), "final_score": 7},
             {"candidate_id": "c", "window": (40, 70), "final_score": 6}]
        kept, dropped = nms(c)
        self.assertEqual([k["candidate_id"] for k in kept], ["a", "c"]); self.assertEqual(dropped[0]["candidate_id"], "b")

    def test_diversity_prefers_variety(self):
        def P(i, ep, who, typ, f): return {"promo_id": i, "segments": [{"episode": ep}], "characters": who, "hook_type": typ,
                                           "final_score": f}
        ps = [P("a", 1, ["ANA", "BEN"], "betrayal", 8.0), P("b", 1, ["ANA", "BEN"], "betrayal", 7.9),
              P("c", 20, ["ZOE", "KAI"], "romance", 7.2)]
        order = [p["promo_id"] for p in rank_diverse(ps)]
        self.assertEqual(order[:2], ["a", "c"])                # near-duplicate 'b' pushed down


if __name__ == "__main__":
    unittest.main()
