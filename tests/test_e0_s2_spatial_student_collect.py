"""New spatial/student observation geometry and complete invocation accounting."""
import unittest
from tools import e0_s2_spatial_student_collect as m


class CaptureControls(unittest.TestCase):
    def roi(self):
        return {'tile_yx': [[y, x] for y in range(23, 28) for x in range(47, 52)],
                'far_controls_yx': [[0, 0], [0, 51]]}

    def test_native_rows_keep_both_real_neighborhoods_and_every_time_slot(self):
        value = m.geometry(self.roi())
        self.assertEqual(len(value['anchors_yx']), 52)
        self.assertEqual(len(value['selected_rows']), 1092)
        self.assertEqual(value['selected_rows'], sorted(set(value['selected_rows'])))
        self.assertEqual(len(value['central_tile_yx']), 25)
        for tau in range(21):
            actual = value['selected_rows'][tau*52:(tau+1)*52]
            self.assertEqual(actual, [tau*1508+y*52+x for y, x in value['anchors_yx']])

    def test_overlapping_or_out_of_grid_roi_cannot_silently_duplicate_evidence(self):
        roi = self.roi(); roi['tile_yx'] = [[y, x] for y in range(12, 17) for x in range(24, 29)]
        with self.assertRaises(ValueError): m.geometry(roi)
        roi = self.roi(); roi['far_controls_yx'][0] = [-1, 0]
        with self.assertRaises(ValueError): m.geometry(roi)

    def test_capture_budget_is_three_unchanged_prefixes_with_no_video_decode(self):
        for part in (0, 1):
            value = m.limits(part)
            self.assertEqual(value['native_full_model_flows'], 12)
            self.assertEqual(value['native_FFN_calls'], 480)
            self.assertEqual(value['raw_postwrite_packets'], 480)
            self.assertEqual(value['operand_row_selections'], 2400)
            self.assertEqual(value['packet_tensor_D2H'], 4320)
            self.assertEqual(value['historical_all_arm_state_byte_checks'], 30)
            self.assertEqual(value['BF16_factor_GEMM'], 640)
            self.assertEqual(value['original_W2_H2D_skips'], 324)
            self.assertEqual(value['video_generations'], 0)
            self.assertEqual(value['VAE_decodes'], 0)

    def test_payload_is_real_typed_52xy_and_fits_each_atomic_file_limit(self):
        actual = sum((8 if k == 'rows' else 2)*__import__('math').prod(shape) for k, shape in m.SHAPES.items())
        self.assertEqual(actual, m.PACKET_BYTES)
        self.assertLess(actual+(16384), 128 << 20)
        self.assertLess(actual*480+(3*60802560)+(64 << 20), m.RESOURCES['new_HF_bytes'])


if __name__ == '__main__': unittest.main()
