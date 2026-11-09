import unittest
from pathlib import Path
from helpers import make_episode_text
from src.parser import parse_episode_text, load_show


class TestParser(unittest.TestCase):
    def test_basic(self):
        ep = parse_episode_text(make_episode_text(3), "episode-003-X.txt")
        self.assertEqual(ep.episode_id, 3)
        self.assertEqual(len(ep.scenes), 1)
        ev = ep.events()
        self.assertEqual(len(ev), 11)
        self.assertEqual(ev[0].speaker, "ANA")
        self.assertEqual(ev[0].tone, "Tense")
        self.assertEqual(ev[0].text, "You lied to me!")          # quotes stripped, wording kept
        self.assertEqual(ev[1].start, 2.5)
        self.assertEqual(ep.flags, [])
        self.assertAlmostEqual(ep.duration, 31.0)

    def test_uppercase_headers(self):                           # episode 6 of the real pack looks like this
        ep = parse_episode_text(make_episode_text(6, upper=True), "episode-006-X.txt")
        self.assertEqual(ep.scenes[0].heading, "INT. HALL - DAY")
        self.assertIsNotNone(ep.scenes[0].end)

    def test_missing_and_malformed_timestamps_are_flagged_not_invented(self):
        txt = make_episode_text(2).replace("Start: (00:02.50) | End: (00:04.00)", "Start: (xx:yy) | End: (00:04.00)")
        txt = txt.replace("Start: (00:07.50) | End: (00:09.00)\n", "")
        ep = parse_episode_text(txt, "episode-002-X.txt")
        types = {f["type"] for f in ep.flags}
        self.assertIn("malformed_timestamp", types)
        self.assertIn("missing_timestamp", types)
        bad = [e for e in ep.events() if e.start is None]
        self.assertTrue(bad)                                      # None, never a fabricated value

    def test_episode_number_mismatch_flag(self):
        ep = parse_episode_text(make_episode_text(4, header_num=9), "episode-004-X.txt")
        self.assertEqual(ep.episode_id, 9)
        self.assertIn("episode_number_mismatch", {f["type"] for f in ep.flags})

    def test_event_outside_scene_range_flag(self):
        ep = parse_episode_text(make_episode_text(1, scene_end=10.0), "episode-001-X.txt")
        self.assertIn("event_outside_scene_range", {f["type"] for f in ep.flags})

    def test_load_show_missing_and_unsupported(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "episode-001-a.txt").write_text(make_episode_text(1), encoding="utf-8")
        (d / "episode-003-c.txt").write_text(make_episode_text(3), encoding="utf-8")
        (d / "notes.pdf").write_bytes(b"%PDF")
        show = load_show(d)
        types = {f["type"] for f in show.flags}
        self.assertIn("missing_episodes", types)
        self.assertIn("unsupported_file", types)
        self.assertEqual([e.episode_id for e in show.episodes], [1, 3])

    def test_real_pack_if_present(self):
        import os
        p = Path(os.environ.get("PROMOFORGE_REAL_PACK", "data/input"))   # optional: point at your own copy
        if not p.exists() or not any(p.rglob("*.txt")):
            self.skipTest("real input pack not available (expected; it is confidential and not in the repo)")
        show = load_show(p)
        self.assertEqual(len(show.episodes), 50)
        self.assertEqual([e.episode_id for e in show.episodes], list(range(1, 51)))


if __name__ == "__main__":
    unittest.main()
