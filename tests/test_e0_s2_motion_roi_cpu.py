"""Capture-coordinate geometry, not a new image/FFN experiment."""
import unittest
import numpy as np
from tools import e0_s2_motion_roi_cpu as m


class GeometryControls(unittest.TestCase):
    def test_real_contiguous_neighborhood_and_far_controls(self):
        heat = np.zeros((29, 52)); heat[10:15, 20:25] = 2
        result = m.anchors(heat, np)
        self.assertEqual(result['roi_center_yx'], [12, 22])
        self.assertEqual(result['roi_mean_flow_pixels_per_adjacent_frame'], 2)
        self.assertEqual(len(result['tile_yx']), 25)
        self.assertEqual(len(set(map(tuple, result['anchors_yx']))), 27)
        self.assertEqual(len(result['selected_rows']), 567)
        self.assertEqual(result['selected_rows'], sorted(set(result['selected_rows'])))
        for tau in range(21):
            self.assertEqual([v % 1508 for v in result['selected_rows'][tau*27:(tau+1)*27]],
                             [y*52+x for y, x in result['anchors_yx']])

    def test_uniform_camera_flow_has_deterministic_ties_not_fake_foreground(self):
        result = m.anchors(np.ones((29, 52)), np)
        self.assertEqual(result['roi_center_yx'], [2, 2])
        self.assertEqual(result['roi_mean_flow_pixels_per_adjacent_frame'], 1)
        self.assertEqual(len(result['far_controls_yx']), 2)

    def test_invalid_flow_geometry_is_rejected(self):
        for heat in (np.zeros((28, 52)), np.full((29, 52), np.nan), -np.ones((29, 52))):
            with self.assertRaises(ValueError): m.anchors(heat, np)


if __name__ == '__main__': unittest.main()
