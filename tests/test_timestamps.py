import unittest
from helpers import *  # noqa
from src.timeutil import ts_to_sec, sec_to_ts, duration


class TestTimestamps(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual(ts_to_sec("00:00.00"), 0)
        self.assertEqual(ts_to_sec("01:23.45"), 83.45)
        self.assertEqual(ts_to_sec("02:18.00"), 138.0)
        self.assertEqual(ts_to_sec("1:00:05.5"), 3605.5)
        self.assertEqual(ts_to_sec("12.5"), 12.5)

    def test_roundtrip_and_carry(self):
        for x in (0, 1.25, 59.99, 60.0, 83.45, 199.99):
            self.assertAlmostEqual(ts_to_sec(sec_to_ts(x)), x, places=2)
        self.assertEqual(sec_to_ts(59.999), "01:00.00")     # carry, never '00:60.00'

    def test_malformed(self):
        for bad in ("abc", "01:75.00", "", "1:2:3:4", "01:xx.00"):
            with self.assertRaises(ValueError):
                ts_to_sec(bad)

    def test_ordering_and_duration(self):
        self.assertEqual(duration(10, 35.5), 25.5)
        with self.assertRaises(ValueError):
            duration(10, 5)
        self.assertLess(ts_to_sec("00:59.99"), ts_to_sec("01:00.00"))


if __name__ == "__main__":
    unittest.main()
