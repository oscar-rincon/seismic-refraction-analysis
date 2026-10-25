import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from utils.plotting import plot_1d_profile, vp_class_layers


class Profile1DTest(unittest.TestCase):
    def test_layers_follow_vp_classes(self):
        z = np.arange(0.0, 6.0, 1.0)
        layers = vp_class_layers(z, [300, 400, 800, 900, 1600, 1600])
        self.assertEqual([l["class"] for l in layers], [0, 1, 3])
        self.assertEqual([l["bottom_m"] for l in layers[:-1]], [1.5, 3.5])
        self.assertEqual(layers[1]["top_m"], 1.5)

    def test_plot_has_vp_vs_and_boundaries(self):
        z = np.linspace(0, 8, 9)
        vp = pd.DataFrame({"depth_m": z, "vp_m_s": np.linspace(300, 1200, 9),
                           "vp_p10_m_s": np.linspace(250, 1100, 9),
                           "vp_p90_m_s": np.linspace(350, 1300, 9)})
        vs = pd.DataFrame({"depth_m": z, "vs_m_s": np.linspace(150, 400, 9)})
        fig, layers = plot_1d_profile(vp, vs)
        ax_v, ax_m = fig.axes
        self.assertEqual(len(ax_v.get_lines()), 2 + len(layers) - 1)
        boundary_texts = [t.get_text() for t in ax_m.texts if t.get_text().endswith(" m")]
        self.assertEqual(len(boundary_texts), len(layers) - 1)


if __name__ == "__main__":
    unittest.main()
