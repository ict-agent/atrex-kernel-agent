from __future__ import annotations

import unittest

from tools.bench_ascend_bandwidth import _tb_s


class AscendBandwidthCalculationTests(unittest.TestCase):
    def test_copy_counts_one_read_and_one_write(self) -> None:
        size = 256 * 1024 * 1024
        expected = 1.2642096666460558
        elapsed = 2 * size * 50 / (expected * 1e12)

        self.assertAlmostEqual(_tb_s(size, 50, elapsed), expected)

    def test_invalid_measurement_fails_closed(self) -> None:
        for args in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                _tb_s(*args)


if __name__ == "__main__":
    unittest.main()
