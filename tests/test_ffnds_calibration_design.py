"""Calibration method v2 regression tests; these exercise real logic on synthetic inputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ffnds_calibration_design import (
    validate_design, coverage, source_support_diagnostic, diagnose
)


class CalibrationDesignTests(unittest.TestCase):
    def setUp(self):
        import json
        self.plan = json.loads((Path(__file__).resolve().parents[1] /
            "config/calibration_design_v2_20261027.json").read_text(encoding="utf-8"))

    def test_design_counts_and_reference_arm(self):
        self.assertEqual(validate_design(self.plan), [])
        arms = {x["id"]: x for x in self.plan["calibration_arms"]}
        self.assertEqual(arms["C12_R512"]["source_count"] * arms["C12_R512"]["rows_per_source"], 6144)
        self.assertEqual(arms["C24_R256"]["source_count"] * arms["C24_R256"]["rows_per_source"], 6144)
        self.assertEqual(arms["C48_R128"]["source_count"] * arms["C48_R128"]["rows_per_source"], 6144)

    def test_coverage_and_image_split(self):
        bank = {"schema": "ffnds.source.bank.v1", "sources": [
            {"source_id": str(i), "image_id": str(i), "scene_stratum": "i2v_subject" if i < 12 else "i2v_background",
             "role": "CALIBRATION"} for i in range(24)]}
        result = coverage(self.plan, bank)
        arms = {x["arm_id"]: x for x in result["arms"]}
        self.assertEqual(arms["C24_R256"]["status"], "SOURCE_CANDIDATES_PRESENT")
        self.assertEqual(arms["C48_R128"]["status"], "AWAITING_NEW_STRATIFIED_SOURCES")
        self.assertEqual(arms["C24_R256"]["native_fit"], "NOT_RUN")
        bank["sources"].append({"source_id": "fresh", "image_id": "0",
            "role": "FROZEN_CONFIRMATION", "scene_stratum": "i2v_subject"})
        self.assertEqual(coverage(self.plan, bank)["status"], "INVALID")

    def test_jackknife_support_detects_source_shift(self):
        a = [0.] * 216
        b = [0.] * 216
        a[0], a[1] = 9., 8.
        b[0], b[2] = 9., 20.
        records = [
            {"source_id": "A", "image_id": "A", "group_energy": a},
            {"source_id": "B", "image_id": "B", "group_energy": b},
        ]
        r = source_support_diagnostic(records, keep_groups=1, bootstrap=100, seed=1)
        self.assertLess(r["leave_one_cluster_out_jaccard"]["mean"], 1)
        self.assertEqual(r["independent_image_clusters"], 2)
        self.assertEqual(r["calibration_sufficiency"], "NOT_ESTABLISHED_BY_LOCAL_DIAGNOSTICS")

    def test_multiple_prompts_same_image_one_independent_cluster(self):
        a = [0.] * 216
        a[1] = 2
        rows = [{"source_id": "A1", "image_id": "A", "group_energy": a},
                {"source_id": "A2", "image_id": "A", "group_energy": a}]
        result = source_support_diagnostic(rows, keep_groups=1)
        self.assertEqual(result["independent_image_clusters"], 1)
        self.assertEqual(result["leave_one_cluster_out_jaccard"], None)

    def test_diagnostics_evidence_requires_actual_native(self):
        with self.assertRaisesRegex(ValueError, "actual native"):
            diagnose({"schema": "ffnds.calibration.source_statistics.v2",
                      "evidence_kind": "PLANNED", "records": []}, keep_groups_by_step=[54,108,108,162])
        a = [0.] * 216
        a[0] = 1
        p = {"schema": "ffnds.calibration.source_statistics.v2",
             "evidence_kind": "ACTUAL_NATIVE_OBSERVATION",
             "records": [{"source_id": "A", "image_id": "A", "source_state": "teacher_input",
                          "step": 0, "block": 0, "group_energy": a,
                          "statistic_semantics": "native_gate_times_W2_group_contribution_l2_squared"}]}
        r = diagnose(p, keep_groups_by_step=[54,108,108,162])
        self.assertEqual(r["new_video_scores"], "NOT_RUN")

if __name__ == "__main__":
    unittest.main()
