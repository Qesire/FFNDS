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

    def test_equal_source_conditions_inside_shared_image_cluster(self):
        # Four repetitions for one prompt must not outweigh a second prompt on the same image.
        rows = [{"source_id": "A", "image_id": "shared", "seed": i,
                 "metric": "imaging_quality", "dense": 1, "candidate": .5}
                for i in range(4)]
        rows += [{"source_id": "B", "image_id": "shared", "seed": 1,
                  "metric": "imaging_quality", "dense": 1, "candidate": 1.0},
                 {"source_id": "C", "image_id": "other", "seed": 1,
                  "metric": "imaging_quality", "dense": 1, "candidate": 1.0}]
        score = paired_report(rows, replicates=0)["metrics"]["imaging_quality"]
        self.assertEqual(score["independent_source_clusters"], 2)
        self.assertEqual(score["actual_sample_records"], 6)
        self.assertAlmostEqual(score["relative_percent"], -12.5)
        shared = next(x for x in score["source_results"] if x["cluster"] == "shared")
        self.assertEqual(shared["n_source_conditions"], 2)

    def test_seed_and_sample_index_are_joint_identity(self):
        rows = [{"source_id": "A", "image_id": "imageA", "seed": s, "sample_index": 0,
                 "metric": "imaging_quality", "dense": 1, "candidate": .9}
                for s in (41, 42)]
        score = paired_report(rows, replicates=0)["metrics"]["imaging_quality"]
        self.assertEqual(score["actual_sample_records"], 2)
        with self.assertRaisesRegex(ValueError, "multiple image clusters"):
            paired_report(rows + [dict(rows[0], metric="aesthetic_quality",
                                       image_id="another")])

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
