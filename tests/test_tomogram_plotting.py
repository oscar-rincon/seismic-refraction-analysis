from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg

from utils.plotting import (
    plot_material_classes, plot_tomogram, plot_traveltime_fit, plot_velocity_contours,
)


class TomogramPlottingTests(unittest.TestCase):
    def test_material_classes_have_fixed_thresholds_colors_and_all_labels(self):
        from matplotlib.colors import to_rgba

        mesh = pg.Mesh(2)
        nodes = [mesh.createNode(x, y, 0) for x, y in
                 [(0, 0), (5, 0), (5, -5), (0, -5)]]
        mesh.createTriangle(nodes[0], nodes[1], nodes[2])
        mesh.createTriangle(nodes[0], nodes[2], nodes[3])
        for velocity in ([300.0, 1700.0], [700.0, 700.0], [2000.0, 2500.0]):
            manager = SimpleNamespace(paraDomain=mesh, velocity=np.array(velocity))
            ax = plot_material_classes(manager, xlim=(0, 5))
            try:
                ax.figure.canvas.draw()
                filled = ax.collections[0]
                np.testing.assert_array_equal(filled.levels[:4], [0, 500, 1000, 1500])
                samples = [499.9, 500, 1000, 1500, 3000]
                expected = ["#eadbb8", "#c8a06e", "#8c6a4a", "#7d8186", "#7d8186"]
                np.testing.assert_allclose(
                    filled.to_rgba(samples), [to_rgba(c) for c in expected])
                self.assertEqual(
                    [t.get_text() for t in ax.get_legend().get_texts()],
                    ["< 500", "500–1000", "1000–1500", "≥ 1500"])
                self.assertEqual(len(ax.figure.axes), 1)
                self.assertEqual(ax.get_title(), "")
                self.assertEqual(len(ax.texts), 0)
                np.testing.assert_array_equal(manager.velocity, velocity)
            finally:
                plt.close(ax.figure)
        manager.velocity = np.array([0.0, 700.0])
        with self.assertRaises(ValueError):
            plot_material_classes(manager)

    def test_fit_connects_predictions_sorted_by_receiver_not_observations(self):
        data = pg.DataContainer()
        data.registerSensorIndex("s")
        data.registerSensorIndex("g")
        for x in (0, 5, 10):
            data.createSensor([x, 0])
        data.resize(2)
        data["s"] = [0, 0]
        data["g"] = [2, 1]
        data["t"] = [0.02, 0.01]
        manager = SimpleNamespace(inv=SimpleNamespace(response=np.array([0.021, 0.012])))
        ax = plot_traveltime_fit(manager, data)
        try:
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [5, 10])
            np.testing.assert_allclose(ax.lines[0].get_ydata(), [12, 21])
            np.testing.assert_allclose(ax.collections[0].get_offsets(), [[5, 10], [10, 20]])
            manager.inv.response = [0.01]
            with self.assertRaises(ValueError):
                plot_traveltime_fit(manager, data)
        finally:
            plt.close(ax.figure)

    def test_contours_use_requested_ranges_and_labeled_colorbar(self):
        mesh = pg.Mesh(2)
        nodes = [mesh.createNode(x, y, 0) for x, y in
                 [(0, 0), (5, 0), (5, -5), (0, -5)]]
        mesh.createTriangle(nodes[0], nodes[1], nodes[2])
        mesh.createTriangle(nodes[0], nodes[2], nodes[3])
        velocity = np.array([300.0, 1500.0])
        manager = SimpleNamespace(paraDomain=mesh, velocity=velocity)
        levels = [0, 500, 1000, 1500, 2000]
        ax, colorbar = plot_velocity_contours(manager, levels)
        try:
            ax.figure.canvas.draw()
            np.testing.assert_array_equal(colorbar.mappable.levels, levels)
            self.assertEqual(len(ax.texts), 0)
            self.assertEqual(ax.get_title(), "")
            self.assertEqual(colorbar.ax.get_xlabel(), "Vp (m/s)")
            np.testing.assert_array_equal(velocity, [300, 1500])
            with self.assertRaises(ValueError):
                plot_velocity_contours(manager, [1000, 500])
        finally:
            plt.close(ax.figure)

    def test_smooth_plot_has_colorbar_without_edges_and_preserves_model(self):
        mesh = pg.Mesh(2)
        nodes = [mesh.createNode(x, y, 0) for x, y in
                 [(0, 0), (5, 0), (5, -5), (0, -5)]]
        mesh.createTriangle(nodes[0], nodes[1], nodes[2])
        mesh.createTriangle(nodes[0], nodes[2], nodes[3])
        velocity = np.array([300.0, 1500.0])
        original = velocity.copy()
        manager = SimpleNamespace(paraDomain=mesh, velocity=velocity)
        ax, colorbar = plot_tomogram(
            manager, smooth=True, rays=False, height=4.5, title="Test")
        try:
            ax.figure.canvas.draw()
            self.assertEqual(len(ax.collections), 1)
            self.assertEqual(len(ax.collections[0].get_edgecolors()), 0)
            self.assertEqual(colorbar.ax.get_xlabel(), r"$V_p$ (m/s)")
            self.assertGreater(colorbar.ax.get_position().y0, 0)
            self.assertLess(colorbar.ax.get_position().y1, ax.get_position().y0)
            values = np.asarray(colorbar.mappable.get_array())
            self.assertGreaterEqual(values.min(), velocity.min())
            self.assertLessEqual(values.max(), velocity.max())
            np.testing.assert_array_equal(velocity, original)
        finally:
            plt.close(ax.figure)


if __name__ == "__main__":
    unittest.main()
