"""Source-level inference with repeated seeds rather than pseudoreplication."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_cluster_metric_report import paired_report


class ClusterMetricTests(unittest.TestCase):
    def test_cluster_equal_weight_despite_unequal_seed_counts(self):
        rows = [{"source_id": "A", "image_id": "imageA", "seed": i,
                 "metric": "imaging_quality", "dense": 1, "candidate": 0.8}
                for i in range(5)]
        rows.append({"source_id": "B", "image_id": "imageB", "seed": 0,
                     "metric": "imaging_quality", "dense": 1, "candidate": 1})
        r = paired_report(rows, replicates=10)["metrics"]["imaging_quality"]
        self.assertEqual(r["independent_source_clusters"], 2)
        self.assertEqual(r["actual_sample_records"], 6)
        self.assertAlmostEqual(r["relative_percent"], -10)

    def test_historical_pairs_and_zero_baseline(self):
        rows = [{"source_group": "subject:0", "dimension": "motion_smoothness",
                 "D_Q": "0.0", "candidate": "0.0"}]
        r = paired_report(rows)["metrics"]["motion_smoothness"]
        self.assertIsNone(r["relative_percent"])
        self.assertEqual(r["relative_percent_status"], "UNDEFINED_ZERO_BASELINE")
        self.assertIsNone(r["delta_bootstrap_95_ci"])

    def test_duplicate_pair_and_missing_value_are_errors(self):
        row = {"source_id": "A", "metric": "imaging_quality", "dense": 1, "candidate": .8}
        with self.assertRaises(ValueError):
            paired_report([row, row])
        with self.assertRaises((ValueError, TypeError)):
            paired_report([dict(row, candidate="NOT_RUN")])


if __name__ == "__main__":
    unittest.main()
