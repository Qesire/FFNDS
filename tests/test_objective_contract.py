"""Guard primary FFNDS objective against silent reinterpretation as static or local-only recovery."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def record(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class ObjectiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = record("research/OBJECTIVE_CONTRACT.json")
        cls.current = record("research/CURRENT_DECISION.json")
        cls.triage = record("research/20261021/method_triage.json")
        cls.protocol = record("research/20261021/R11T1_matched_objective_protocol.json")

    def test_canonical_contract_referenced_without_releasing_model_jobs(self):
        self.assertEqual(self.current["objective_contract"], "research/OBJECTIVE_CONTRACT.json")
        self.assertEqual(self.triage["policy"]["objective_contract"], self.current["objective_contract"])
        if self.current['eligible_model_method_ids'] == ['R17E0_FRESH23_NINE_METRICS_AND_SYSTEM_TIMING']:
            registered = record(self.current['registered_fresh_quality']['protocol'])
            authorization = record(registered['authorization']['path'])
            self.assertEqual(authorization['status'], 'EXPLICIT_EXACT_SCOPE_USER_AUTHORIZATION')
            self.assertEqual(authorization['method_id'], registered['id'])
            self.assertEqual(registered['caps'], {'full_generations': 145, 'metric_evaluator_calls': 45,
                                                 'standalone_FFN_forwards': 0, 'training_steps': 0,
                                                 'weight_factorizations': 0})
            self.assertFalse(registered['automatic_followup'])
            self.assertFalse(registered['new_training_or_parameter_selection'])
        elif self.current['eligible_model_method_ids'] == ['R16V0_FULL160_UB1_DENSE_ROW_DISPATCH']:
            registered = record(self.current['registered_row_dispatch']['protocol'])
            authorization = record(registered['authorization']['path'])
            self.assertEqual(authorization['status'], 'EXPLICIT_EXACT_SCOPE_USER_AUTHORIZATION')
            self.assertEqual(authorization['method_id'], registered['id'])
            self.assertEqual(registered['caps'], {'full_generations': 5, 'standalone_FFN_forwards': 164,
                                                 'weight_factorizations': 0, 'training_steps': 0})
            self.assertFalse(registered['automatic_followup'])
            self.assertFalse(registered['reset_between_steps'])
        elif self.current['eligible_model_method_ids'] == ['R15V0_MEAN_CONTRAST_PAIRED_ROLLOUT']:
            registered = record(self.current['registered_rollout']['protocol'])
            authorization = record(registered['authorization']['path'])
            self.assertEqual(authorization['status'], 'EXPLICIT_EXACT_SCOPE_USER_AUTHORIZATION')
            self.assertEqual(authorization['method_id'], registered['id'])
            self.assertEqual(registered['caps'], {
                'full_generations': 2, 'offline_weight_factorizations': 2, 'training_steps': 0})
            self.assertFalse(registered['automatic_followup'])
            self.assertFalse(registered['reset_between_steps'])
        elif self.current['eligible_model_method_ids']:
            self.assertIn(self.current['eligible_model_method_ids'],
                          [['R12E1_D0_EQUAL_BUDGET_NATIVE_VALUE'], ['R14N0_COMPLETE_STRIPE_NATIVE_VALUE']])
            registered = record(self.current['registered_diagnostic']['protocol'])
            authorization = record(registered['authorization']['path'])
            self.assertEqual(authorization['status'], 'EXPLICIT_USER_AUTHORIZATION')
            self.assertFalse(registered['fresh_quality_acceptance'])
            self.assertFalse(registered['automatic_followup_authorization'])
        self.assertIn("NO_EXECUTION_AUTHORIZATION", self.contract["status"])
        self.assertEqual(self.current["latest_objective_alignment"]["eligibility_effect"].split(";")[0], "NONE")

    def test_dynamic_objective_is_distinct_from_static_ub1(self):
        goal = self.contract["primary_scientific_goal"].lower()
        self.assertIn("dynamic ffn sparsity", goal)
        self.assertIn("input-conditioned", goal)
        self.assertIn("pre", self.contract["target_mechanism"]["timing_requirement"].lower().replace("before", "pre"))
        self.assertIn("static", self.contract["current_evidence_interpretation"]["UB1"].lower())
        self.assertIn("D-Q", self.contract["target_system"]["baseline"])
        self.assertIn("undistilled O", self.contract["target_system"]["explicit_non_equivalence"])

    def test_legacy_registered_acceptance_preserves_old_numbers_and_requires_fresh_quality(self):
        gates = self.contract["existing_joint_acceptance"]
        self.assertEqual(gates["main_gemm_MAC_fraction_max"], 0.5)
        self.assertEqual(gates["complete_original_FFN_speedup_min"], 1.5)
        self.assertEqual(gates["per_axis_relative_quality_drop_max"], 0.02)
        self.assertEqual(gates["quality_axes"], ["aesthetic", "imaging", "motion"])
        self.assertIn("historical registered", gates["scope"].lower())
        self.assertIn("not a universal", gates["scope"].lower())
        self.assertIn("free-running", gates["quality_population"])
        self.assertIn("routing", gates["cost_scope"].lower())
        self.assertEqual(len(self.contract["evidence_hierarchy"]), 5)

    def test_r11_is_only_fixed_input_diagnostic_and_closed(self):
        self.assertFalse(self.protocol["fresh_quality_acceptance"])
        self.assertEqual(self.protocol["positions"], [[3, 39]])
        self.assertIn("not free student rollout", self.protocol["inputs"]["trajectory"])
        self.assertEqual(
            self.current["closed_method_statuses"]["R11T1_MATCHED_NATIVE_HEAD_OBJECTIVE"],
            "CLOSED_FIXED20_MATCHED_HEAD_OBJECTIVE_REJECTED",
        )
        self.assertIn("NOT quality", self.current["latest_objective_alignment"]["R11T1_scope"])
        self.assertIn("DQ trajectory", self.current["latest_objective_alignment"]["quality_scope"])

    def test_documentation_retains_provenance_and_boundaries(self):
        note = (ROOT / "docs/65_研究目标层级与R11解释修正_20261022.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        handoff = (ROOT / "research/20261021/HANDOFF_R11T1.md").read_text(encoding="utf-8")
        for body in (note, readme, agents, handoff):
            self.assertTrue("动态" in body or "dynamic" in body.lower())
            self.assertTrue("R11" in body or "flow" in body)
        self.assertIn("不是新的实验", note)
        self.assertIn("OBJECTIVE_CONTRACT.json", agents)
        self.assertIn("OBJECTIVE_CONTRACT.json", readme)
        self.assertIn("OBJECTIVE_CONTRACT.json", handoff)


if __name__ == "__main__":
    unittest.main()
