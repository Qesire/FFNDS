from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_saved_flow_partitions import summarize_partitions


class SavedFlowPartitionTests(unittest.TestCase):
    def test_zero_video_headroom_can_hide_opposite_region_preferences(self):
        row = {"whole_loss": {"A": 2., "B": 2.},
               "frame_losses": {"A": [2.], "B": [2.]},
               "native_token_losses": {"A": [0., 2.], "B": [2., 0.]}}
        result = summarize_partitions([row])
        self.assertEqual(result["post_action_mosaic_total_losses"], {"source": 2., "frame": 2., "native_token": 0.})
        self.assertEqual(result["mosaic_reduction_over_best_fixed_fraction"]["source"], 0.)

    def test_unmatched_aggregation_is_rejected(self):
        row = {"whole_loss": {"A": 2., "B": 2.},
               "frame_losses": {"A": [2.], "B": [2.]},
               "native_token_losses": {"A": [0., 1.], "B": [2., 0.]}}
        with self.assertRaisesRegex(ValueError, "sum to original"):
            summarize_partitions([row])


if __name__ == "__main__":
    unittest.main()
