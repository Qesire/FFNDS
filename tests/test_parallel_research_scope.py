"""Pure-static research-entrypoint and historical permission regression tests."""
import json
from pathlib import Path
import unittest

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
        self.assertIn("PAUSED", a["working_mode"])
        old = c["existing_joint_acceptance"]
        self.assertEqual([old["main_gemm_MAC_fraction_max"],
                          old["complete_original_FFN_speedup_min"],
                          old["per_axis_relative_quality_drop_max"]], [0.5, 1.5, 0.02])

if __name__ == "__main__":
    unittest.main()
