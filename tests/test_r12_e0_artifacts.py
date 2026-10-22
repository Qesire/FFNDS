from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_r12_e0_artifacts import BASE, read, validate


class R12EvidenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.records = [read(p) for p in [BASE + "R12_E0_operand_audit.json", BASE + "R12_E0_flow_partitions.json",
                        BASE + "R12_E1_native_value_DRAFT.json", BASE + "R12_E0_runtime_reconstruction.json",
                        "research/CURRENT_DECISION.json", "research/20261021/method_triage.json"]]

    def test_completed_records_are_consistent(self):
        self.assertEqual(validate(*self.records), [])

    def test_output_mosaic_cannot_be_promoted_to_native_result(self):
        self.records[1]["evidence_level"] = "NATIVE_DYNAMIC_PASS"
        self.assertIn("output stitching relabeled dynamic execution", validate(*self.records))

    def test_cpu_stage_cannot_fabricate_action_observations(self):
        self.records[0]["counts"]["R12_counterfactual_action_flows"] = 12
        self.assertIn("scoped counts or unobserved R12 actions changed", validate(*self.records))

    def test_draft_cannot_self_grant_execution(self):
        self.records[2]["status"] = "ELIGIBLE"
        self.records[4]["eligible_model_method_ids"] = [self.records[2]["id"]]
        errors = validate(*self.records)
        self.assertIn("draft status changed", errors)
        self.assertIn("E0 cannot grant model eligibility", errors)


if __name__ == "__main__":
    unittest.main()
