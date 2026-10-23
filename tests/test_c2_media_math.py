"""Known pixel/PSNR reductions for C2; synthetic scalars, no video/model."""
from pathlib import Path
import math
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from review_c2_native_videos import pixel_summary


class C2MediaMath(unittest.TestCase):
    def test_mean_frame_and_pooled_psnr_are_distinct(self):
        frames = [10 * math.log10(255 ** 2), 10 * math.log10(255 ** 2 / 9)]
        r = pixel_summary(10 * 12, 4 * 12, frames, 12)
        self.assertEqual(r['pooled_RGB_MSE_0_255'], 5)
        self.assertEqual(r['pooled_RGB_MAE_0_255'], 2)
        self.assertAlmostEqual(r['mean_frame_PSNR_dB'], sum(frames) / 2)
        self.assertAlmostEqual(r['pooled_MSE_PSNR_dB'], 10 * math.log10(255 ** 2 / 5))
        self.assertNotEqual(r['pooled_MSE_PSNR_dB'], r['mean_frame_PSNR_dB'])

    def test_exact_frames_are_not_silently_dropped_from_psnr(self):
        r = pixel_summary(9 * 12, 3 * 12, [None, 10 * math.log10(255 ** 2 / 9)], 12)
        self.assertIsNone(r['mean_frame_PSNR_dB'])
        self.assertEqual(r['exact_match_frames'], 1)
        self.assertAlmostEqual(r['pooled_MSE_PSNR_dB'], 10 * math.log10(255 ** 2 / 4.5))
        exact = pixel_summary(0, 0, [None, None], 12)
        self.assertIsNone(exact['pooled_MSE_PSNR_dB'])
        self.assertEqual(exact['exact_match_frames'], 2)


if __name__ == '__main__': unittest.main()
