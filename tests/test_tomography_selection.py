import unittest

import numpy as np
import pygimli as pg

from utils.tomography_selection import select_model, spatial_folds, subset


class TomographySelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = pg.DataContainer()
        self.data.registerSensorIndex("s")
        self.data.registerSensorIndex("g")
        for x in (0, 5, 10, 15):
            self.data.createSensor([x, 0])
        self.data.resize(6)
        self.data["s"] = [0, 1, 0, 2, 1, 3]
        self.data["g"] = [1, 0, 2, 0, 3, 1]
        self.data["t"] = [0.01] * 6
        self.data["valid"] = [1] * 6

    def test_reciprocals_are_never_split(self):
        folds = spatial_folds(self.data)
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(folds[2], folds[3])
        self.assertEqual(folds[4], folds[5])
        self.assertEqual(set(folds), {0, 1, 2})

    def test_subset_preserves_input_and_sensor_indices(self):
        mask = np.array([True, False, False, True, False, True])
        selected = subset(self.data, mask)
        self.assertEqual(self.data.size(), 6)
        self.assertEqual(selected.size(), 3)
        self.assertEqual(selected.sensorCount(), self.data.sensorCount())
        np.testing.assert_array_equal(selected["s"], np.asarray(self.data["s"])[mask])
        np.testing.assert_array_equal(selected["g"], np.asarray(self.data["g"])[mask])

    def test_invalid_observations_raise(self):
        self.data["t"] = [0] * 6
        with self.assertRaises(ValueError):
            select_model(self.data)

    def test_invalid_fold_counts_raise(self):
        for count in (1, 4):
            with self.assertRaises(ValueError):
                spatial_folds(self.data, count)


if __name__ == "__main__":
    unittest.main()
