"""Spatial evidence must use true token coordinates, with no frame wrap."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_spatial_residual_coherence import coordinates, far_donor, pairs


class SpatialResidualCoordinateTests(unittest.TestCase):
    def test_native_neighbours_do_not_wrap_rows_or_frames(self):
        selected = [0, 1, 7, 8, 9, 23, 24, 25]
        actual = {(selected[a], selected[b]) for a, b in pairs(selected, [2, 3, 8], 'near')}
        self.assertEqual(actual, {(0, 1), (0, 8), (1, 9), (8, 9), (24, 25)})
        self.assertNotIn((7, 8), actual)
        self.assertNotIn((23, 24), actual)

    def test_horizontal_stripes_are_actual_groups_not_sampled_offsets(self):
        selected = [0, 1, 3, 4, 7, 8, 11, 24]
        actual = {(selected[a], selected[b]) for a, b in pairs(selected, [2, 3, 8], 'stripe')}
        self.assertEqual(actual, {(0, 1), (0, 3), (1, 3), (4, 7), (8, 11)})
        with self.assertRaises(ValueError):
            pairs([1, 0], [2, 3, 8], 'near')

    def test_far_control_stays_in_target_frame_and_excludes_neighbours(self):
        selected = [0, 1, 7, 23, 24, 47]
        donor = far_donor(selected, [2, 3, 8], 0)
        self.assertEqual(selected[donor], 23)
        self.assertEqual(coordinates(selected[donor], [2, 3, 8])[0], 0)
        self.assertIsNone(far_donor([0, 1, 24], [2, 3, 8], 0))

    def test_equal_budget_layout_counts_the_odd_grid_boundary(self):
        frames, height, width = [21, 29, 52]
        m = frames * height * width
        stripes = frames * height * (width // 4)
        squares = frames * ((height + 1) // 2) * ((width + 1) // 2)
        self.assertEqual(stripes, 7917)
        self.assertEqual(squares, 8190)
        self.assertEqual(.25 + stripes / m, .5)
        self.assertGreater(.25 + squares / m, .5)


if __name__ == '__main__':
    unittest.main()
