import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import pandas as pd

from utils.pick_refinement import aic_candidate, select_refined_picks


class PickRefinementTests(unittest.TestCase):
    def test_only_stable_candidates_replace_references_and_manual_picks_win(self):
        gather = SimpleNamespace(n_receivers=2, time=np.arange(400) * 0.000125)
        original = [np.array([0.010, 0.020])]
        candidate = [np.array([0.008, 0.015])]
        review = pd.DataFrame([
            dict(shot_index=0, receiver_index=0, original_ms=10,
                 review_flags="visual confirmation required"),
            dict(shot_index=0, receiver_index=1, original_ms=20,
                 review_flags="weak variance contrast (<3)"),
        ])
        with patch("utils.pick_refinement.refine_line", return_value=(candidate, review.copy())):
            selected, result = select_refined_picks([gather], original)
        np.testing.assert_allclose(selected[0], [0.008, 0.020])
        np.testing.assert_allclose(original[0], [0.010, 0.020])
        self.assertEqual(result.needs_review.tolist(), [False, True])
        with patch("utils.pick_refinement.refine_line", return_value=(candidate, review.copy())):
            selected, result = select_refined_picks(
                [gather], original, overrides={(0, 0): np.nan, (0, 1): 0.017})
        self.assertTrue(np.isnan(selected[0][0]))
        self.assertEqual(selected[0][1], 0.017)
        self.assertFalse(result.needs_review.any())
        with patch("utils.pick_refinement.refine_line", return_value=(candidate, review.copy())):
            with self.assertRaises(ValueError):
                select_refined_picks([gather], original, overrides={(0, 1): -1})

    def test_known_variance_onset_and_amplitude_invariance(self):
        rng = np.random.default_rng(42)
        trace = np.concatenate([rng.normal(0, 0.01, 160), rng.normal(0, 1, 240)])
        original = trace.copy()
        time, edge = aic_candidate(trace, 0.000125, 0.022)
        self.assertLess(abs(time - 0.020), 0.001)
        self.assertFalse(edge)
        scaled, _ = aic_candidate(-trace * 100, 0.000125, 0.022)
        self.assertEqual(time, scaled)
        np.testing.assert_array_equal(trace, original)

    def test_no_signal_and_invalid_inputs_raise(self):
        for trace, dt, reference in [
            (np.zeros(400), 0.000125, 0.02),
            (np.ones(400), 0, 0.02),
            (np.ones(400), 0.000125, np.nan),
            (np.ones(400), 0.000125, 1.0),
        ]:
            with self.assertRaises(ValueError):
                aic_candidate(trace, dt, reference)


if __name__ == "__main__":
    unittest.main()
