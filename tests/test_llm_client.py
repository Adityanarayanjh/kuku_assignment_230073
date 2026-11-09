import json
import tempfile
import unittest
import helpers  # noqa: puts the project root on sys.path
from src.llm_client import FakeClient, LLMError
from src.schema import validate_evaluation, SchemaError
import config

GOOD = json.dumps({**{k: 7 for k in config.WEIGHTS}, "evidence": [{"dimension": "hook", "quote": "q"}], "decision": "accept"})


class TestClient(unittest.TestCase):
    def mk(self, replies, **kw):
        return FakeClient(replies, sleep=lambda s: None, max_retries=3, **kw)

    def test_retry_on_malformed_json(self):
        c = self.mk(["this is not json", GOOD])
        out = c.generate_json("p", validator=validate_evaluation)
        self.assertEqual(c.calls, 2); self.assertEqual(out["decision"], "accept")
        self.assertEqual(c.errors[0]["type"], "invalid_output")

    def test_retry_on_schema_violation(self):
        bad = json.dumps({"hook_score": 99})
        c = self.mk([bad, GOOD])
        self.assertEqual(c.generate_json("p", validator=validate_evaluation)["decision"], "accept")

    def test_code_fence_is_stripped(self):
        c = self.mk(["```json\n" + GOOD + "\n```"])
        self.assertEqual(c.generate_json("p", validator=validate_evaluation)["decision"], "accept")

    def test_api_error_and_rate_limit_backoff(self):
        waits = []
        c = FakeClient([RuntimeError("429 RESOURCE_EXHAUSTED"), GOOD], sleep=waits.append, max_retries=3)
        c.generate_json("p", validator=validate_evaluation)
        self.assertEqual(len(waits), 1); self.assertGreaterEqual(waits[0], 10)   # rate limit waits longer

    def test_gives_up(self):
        c = self.mk(["x", "y", "z"])
        with self.assertRaises(LLMError):
            c.generate_json("p", validator=validate_evaluation)
        self.assertEqual(len(c.errors), 3)

    def test_cache_prevents_repeat_calls(self):
        d = tempfile.mkdtemp()
        c = self.mk([GOOD], cache_dir=d)
        a = c.generate_json("same prompt", validator=validate_evaluation)
        b = c.generate_json("same prompt", validator=validate_evaluation)
        self.assertEqual(a, b); self.assertEqual(c.calls, 1); self.assertEqual(c.cache_hits, 1)

    def test_raw_responses_saved_separately(self):
        d = tempfile.mkdtemp()
        c = self.mk(["garbage", GOOD], raw_dir=d)
        c.generate_json("p", validator=validate_evaluation)
        import os
        self.assertEqual(len(os.listdir(d)), 2)         # both attempts kept, unedited


if __name__ == "__main__":
    unittest.main()
