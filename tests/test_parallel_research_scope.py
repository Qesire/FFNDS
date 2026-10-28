"""Pure-static research-entrypoint and historical permission regression tests."""
import json
import copy
from pathlib import Path
import unittest
from tools.check_autoresearcher_binding import current_execution_authority_is_consistent

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

class ParallelResearchScopeTests(unittest.TestCase):
    def test_parallel_north_star(self):
        d = load("research/OBJECTIVE_CONTRACT.json")["current_research_goal_20261027"]
        self.assertEqual(set(d["parallel_entries"]), {
            "INFORMATION_AND_REPRESENTATION", "FUNCTION_AND_DYNAMICS",
            "COMPUTATION_AND_EXECUTION"})
        self.assertEqual(set(d["independent_contributions"]), {
            "MECHANISM_SCIENCE", "COMPETITIVE_QUALITY_COST_METHOD"})
        self.assertEqual(d["g13_role"],
                         "SCOPED_FUNCTIONAL_CAUSAL_LOCALIZATION_NOT_GLOBAL_GATE")
        self.assertTrue((ROOT / d["authority_document"]).is_file())
        self.assertTrue((ROOT / "research/20261027/PARALLEL_MECHANISM_EVIDENCE_PORTFOLIO.md").is_file())

    def test_real_behavior_cases_cover_science_without_algorithm(self):
        b = load("research/agent_protocol/DEFAULT_BEHAVIOR_CASES.json")
        by_id = {c["id"]: c for c in b["cases"]}
        self.assertEqual(len(by_id), len(b["cases"]))
        x = b["parallel_research_behavior_extension_20261027"]
        self.assertEqual(set(x["tracks"]), {
            "INFORMATION_AND_REPRESENTATION", "FUNCTION_AND_DYNAMICS",
            "COMPUTATION_AND_EXECUTION"})
        for cid in x["case_ids"]:
            self.assertIn(cid, by_id)
            self.assertTrue(by_id[cid]["expected"])
        self.assertEqual(b["g13_function_first_behavior_extension"]["scope"],
                         "F_LINE_ONLY_NOT_GLOBAL_SEQUENCING")

    def test_independent_claim_template(self):
        template = load("research/method_discussion/GRAPH_CLAIM_TEMPLATE.json")
        self.assertEqual(template["research_track"], "SELECT_I_F_OR_C")
        self.assertEqual(template["contribution_kind"], "SELECT_MECHANISM_OR_METHOD")
        self.assertFalse(template["model_experiment_authorized"])

    def test_pause_and_historical_gates_unchanged(self):
        a = load("research/ACTIVE_RESEARCH.json")
        c = load("research/OBJECTIVE_CONTRACT.json")
        self.assertEqual(a["current_stage"]["new_experiment_authority"], "NONE")
        self.assertTrue(current_execution_authority_is_consistent(a))
        old = c["existing_joint_acceptance"]
        self.assertEqual([old["main_gemm_MAC_fraction_max"],
                          old["complete_original_FFN_speedup_min"],
                          old["per_axis_relative_quality_drop_max"]], [0.5, 1.5, 0.02])

    def test_dynamic_gpu_authority_rejects_old_hard_caps_or_unfrozen_scope(self):
        active = load("research/ACTIVE_RESEARCH.json")
        self.assertTrue(current_execution_authority_is_consistent(active))
        for key, value in (("user_instruction", "  "),
                           ("max_simultaneously_assigned_idle_GPUs", 0),
                           ("max_simultaneously_assigned_idle_GPUs", 5),
                           ("protocol_code_asset_budget_freeze_before_launch", False)):
            with self.subTest(key=key):
                changed = copy.deepcopy(active)
                changed["user_continuation_authorization_20261027"][key] = value
                self.assertFalse(current_execution_authority_is_consistent(changed))

    def test_actual_callable_resource_policy_has_no_numeric_global_ceiling(self):
        active = load("research/ACTIVE_RESEARCH.json")
        self.assertIsNone(active["resource_scheduling_policy_20261028"]["global_gpu_card_cap"])
        self.assertEqual(active["resource_scheduling_policy_20261028"]["mode"],
                         "DYNAMIC_MAX_ACTUALLY_CALLABLE")
        self.assertEqual(active["user_continuation_authorization_20261027"]
                         ["historical_previous_GPU_limit_20261028"], 2)
        self.assertEqual(active["user_continuation_authorization_20261027"]
                         ["max_simultaneously_assigned_idle_GPUs"], 2)  # historical, not active global
        for limit in (1, 2, 4, 8, 16, 64):
            with self.subTest(actual_callable=limit):
                # Actual allocation is external; the active authority contains no cap to compare it against.
                self.assertTrue(current_execution_authority_is_consistent(active))
        for bad_policy in (1, 2, 4, 8):
            changed = copy.deepcopy(active)
            changed["resource_scheduling_policy_20261028"]["global_gpu_card_cap"] = bad_policy
            self.assertFalse(current_execution_authority_is_consistent(changed))

    def test_historical_pause_remains_valid(self):
        active = load("research/ACTIVE_RESEARCH.json")
        active["experiment_authority"]["new_model_experiment_requires_separate_explicit_user_instruction"] = True
        active["experiment_authority"]["all_cpu_gpu_model_experiments"] = "PAUSED_BY_EXPLICIT_USER_REQUEST_20261027"
        active["experiment_authority"]["explicit_user_resume_required"] = True
        active["function_first_structure_localization_20261027"]["model_execution_authority"] = "PAUSED_BY_EXPLICIT_USER_REQUEST_20261027"
        self.assertTrue(current_execution_authority_is_consistent(active))

if __name__ == "__main__":
    unittest.main()
