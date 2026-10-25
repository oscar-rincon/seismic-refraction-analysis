import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from utils import masw
from utils.plotting import plot_1d_vp_vs, plot_1d_vs_profile, vs_class_layers


def _curve(frequencies, velocities):
    f, c = np.asarray(frequencies, float), np.asarray(velocities, float)
    return pd.DataFrame({"frequency_Hz": f, "c_rayleigh_m_s": c, "wavelength_m": c / f})


class MaswProfileTest(unittest.TestCase):
    def setUp(self):
        # 60 m/s at 12 Hz is a 5 m wavelength: shorter than 2 x 5 m spacing.
        self.curve = _curve([12, 15, 20, 25, 30], [60, 450, 420, 400, 330])

    def test_min_wavelength_drops_aliased_picks(self):
        profile = masw.vs_profile(self.curve, min_wavelength=10.0)
        self.assertGreaterEqual(profile["depth_m"].min(), 10.0 / 2.5 - 1e-9)
        self.assertGreater(profile["vs_m_s"].min(), 300)
        flagged = masw.aliased_picks(self.curve, 10.0)
        self.assertEqual(len(flagged), 1)
        self.assertAlmostEqual(flagged["depth_m"].iloc[0], 2.0)

    def test_shot_profiles_are_nan_outside_their_range(self):
        profile = masw.vs_profile(self.curve, min_wavelength=10.0)
        short = _curve([20, 25], [420, 400])
        out = masw.add_shot_profiles(profile, [self.curve, short], min_wavelength=10.0)
        self.assertEqual(out["vs_shot1_m_s"].isna().sum(), 0)
        self.assertGreater(out["vs_shot2_m_s"].isna().sum(), 0)

    def test_layer_averages_ignore_boundaries_outside_profile(self):
        profile = pd.DataFrame({"depth_m": np.linspace(4, 8, 9), "vs_m_s": np.full(9, 400.0)})
        layers = masw.layer_averages(profile, [2.0, 6.0])
        self.assertEqual(list(layers["top_m"]), [4.0, 6.0])
        self.assertFalse(layers["vs_mean_m_s"].isna().any())


class VpVsPlotTest(unittest.TestCase):
    def test_vs_classes_and_combined_ratio(self):
        z_s = np.linspace(4, 8, 5)
        self.assertEqual([l["class"] for l in vs_class_layers(z_s, [150, 200, 400, 400, 800])],
                         [0, 1, 2, 3])
        vp = pd.DataFrame({"depth_m": np.linspace(0, 8, 9), "vp_m_s": np.linspace(400, 1200, 9)})
        vs = pd.DataFrame({"depth_m": z_s, "vs_m_s": np.full(5, 400.0),
                           "vs_shot1_m_s": np.full(5, 380.0)})
        fig, vp_layers, vs_layers, ratio = plot_1d_vp_vs(vp, vs)
        self.assertEqual(len(fig.axes), 4)
        np.testing.assert_allclose(ratio["vp_vs"], np.linspace(800, 1200, 5) / 400)
        self.assertEqual(vs_layers[0]["top_m"], 4.0)
        self.assertIn("no data", [t.get_text() for t in fig.axes[3].texts])

    def test_vs_plot_shows_flagged_picks(self):
        vs = pd.DataFrame({"depth_m": np.linspace(4, 8, 5), "vs_m_s": np.full(5, 400.0)})
        flagged = pd.DataFrame({"depth_m": [2.0], "vs_m_s": [65.0]})
        fig, layers = plot_1d_vs_profile(vs, flagged)
        labels = [l.get_label() for l in fig.axes[0].get_lines()]
        self.assertIn("λ < 2Δx", labels)
        self.assertEqual(len(layers), 1)


class MaterialRangeTest(unittest.TestCase):
    def test_intervals_follow_vp_and_vs_ranges(self):
        from utils.interpretation import interpret_vp_vs

        z = np.linspace(0, 8, 81)
        vp = pd.DataFrame({"depth_m": z, "vp_m_s": np.where(z < 2, 800.0, 1200.0)})
        vs = pd.DataFrame({"depth_m": np.linspace(4, 8, 9), "vs_m_s": np.full(9, 400.0)})
        out = interpret_vp_vs(vp, vs)
        self.assertEqual(list(out["vp_range"]), ["500–1000", "1000–1500", "1000–1500"])
        self.assertEqual(list(out["vs_range"]), ["no data", "no data", "360–760"])
        self.assertAlmostEqual(out["bottom_m"].iloc[0], 1.95)
        self.assertAlmostEqual(out["vp_vs"].iloc[2], 3.0)
        self.assertIn("partly saturated", out["interpretation"].iloc[2])
        self.assertEqual(out["material"].iloc[2], "Dense or moist soil (Vs: very dense)")

        from utils.plotting import plot_material_log
        fig = plot_material_log(vp, vs, out)
        labels = [t.get_text() for t in fig.axes[1].texts]
        self.assertIn("Unsaturated soil / fill", labels)
        self.assertTrue(any("Vp/Vs 3.0" in t for t in labels))


if __name__ == "__main__":
    unittest.main()
