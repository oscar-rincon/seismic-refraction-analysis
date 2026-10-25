import unittest

import numpy as np

from utils.shot_stacking import align_and_stack, estimate_hit_lags, shift_samples


def _wavelet(n, onset, fs=8000.0, f=40.0):
    t = (np.arange(n) - onset) / fs
    w = np.where(t >= 0, np.sin(2 * np.pi * f * t) * np.exp(-t * 60), 0.0)
    return w


class ShotStackingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n, nrec = 1500, 6
        base = np.column_stack([_wavelet(n, 200 + 30 * j) for j in range(nrec)])
        self.lags = [0, 12, -7]
        self.hits = np.stack([
            shift_samples(base, s) + 0.01 * rng.standard_normal(base.shape) for s in self.lags
        ])
        self.base = base

    def test_shift_samples_zero_pads(self):
        x = np.arange(5.0)
        np.testing.assert_array_equal(shift_samples(x, 2), [0, 0, 0, 1, 2])
        np.testing.assert_array_equal(shift_samples(x, -2), [2, 3, 4, 0, 0])

    def test_estimates_trigger_lags(self):
        np.testing.assert_array_equal(estimate_hit_lags(self.hits), self.lags)

    def test_aligned_stack_is_sharper_than_plain_mean(self):
        stack, shifts = align_and_stack(self.hits)
        np.testing.assert_array_equal(shifts, [0, -12, 7])  # aligned to median lag (0)
        plain = self.hits.mean(axis=0)
        err_aligned = np.abs(stack - self.base).max()
        err_plain = np.abs(plain - self.base).max()
        self.assertLess(err_aligned, 0.05)
        self.assertGreater(err_plain, 3 * err_aligned)


if __name__ == "__main__":
    unittest.main()
