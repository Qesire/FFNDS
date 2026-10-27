"""Pure-static tests for original Wan source graph plus literature and mechanism crosswalk."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from graph_agent_bridge import load, validate, inspect, trace, load_crosswalk, validate_crosswalk


class GraphAgentBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load()

    def test_registry_matches_mermaid_and_existing_sources(self):
        result = validate(self.registry)
        self.assertEqual(result["status"], "STATIC_GRAPH_BRIDGE_VALID")
        self.assertGreaterEqual(result["nodes"], 20)

    def test_actual_post_q2_input_distinct_from_gelu(self):
        q = inspect(self.registry, "F08")
        self.assertEqual(q["node"]["availability"], "after_Q2")
        self.assertTrue(any("FULL_HIDDEN" in p for p in q["node"]["evidence"]))
        h = inspect(self.registry, "F06")
        self.assertEqual(h["node"]["availability"], "after_GELU")

    def test_cost_paths_preserve_candidate_and_measured_boundaries(self):
        path = trace(self.registry, "G18", "P12")
        self.assertTrue(path["edges"])
        self.assertIn("not proof", path["warning"])
        # The G5 exact-dense SYS branch is now a valid shorter route. It must
        # not be relabeled a hypothetical approximation to satisfy this test.
        routes = {
            "P16": "conditional_approximation_cost_not_executed",
            "P17": "scoped_measured_system_cost",
            "P19": "scoped_native_local_time_excludes_preparation_and_Q1_gate_residual",
        }
        for start, relation in routes.items():
            route = trace(self.registry, start, "P12")
            self.assertIn("not proof", route["warning"])
            self.assertEqual(route["edges"], [{"from": start, "to": "P12", "relation": relation}])

    def test_g5_separates_local_function_observations_from_dense_system_speed(self):
        kinds = {n["id"]: n["evidence_type"] for n in self.registry["nodes"]}
        self.assertEqual(kinds["P15"], "OBS")
        self.assertEqual(kinds["P16"], "OBS")
        self.assertEqual(kinds["P17"], "SYS")
        self.assertEqual(kinds["P18"], "ALG")
        self.assertIn("G5", self.registry["graph_roles"])

    def test_full_depth_video_diagnostics_remain_open_and_policy_is_not_native_evidence(self):
        kinds={n["id"]:n["evidence_type"] for n in self.registry["nodes"]}
        # Prior open hypotheses were later grounded by completed native observations.
        self.assertEqual(kinds["P23"], "NATIVE-OBS")
        self.assertEqual(kinds["P24"], "NATIVE-OBS")
        self.assertEqual(kinds["P25"], "POLICY")
        self.assertTrue(any(e["from"]=="P23" and e["to"]=="P24"
                            for e in self.registry["edges"]))

    def test_literature_branch_is_not_native_execution(self):
        kinds = {n["id"]: n["evidence_type"] for n in self.registry["nodes"]}
        for i in range(1, 13):
            self.assertEqual(kinds[f"K{i:02d}"], "LIT")
        self.assertIn("G7", self.registry["graph_roles"])
        self.assertEqual(self.registry["literature_mapping_v1"]["model_execution_count"], 0)
        path = trace(self.registry, "K07", "F08R")
        self.assertTrue(path["edges"])
        self.assertIn("not proof", path["warning"])

    def test_literature_g8_preserves_paper_execution_boundaries(self):
        kinds = {n["id"]: n["evidence_type"] for n in self.registry["nodes"]}
        for i in range(13, 28):
            self.assertEqual(kinds[f"K{i:02d}"], "LIT")
        self.assertIn("G8", self.registry["graph_roles"])
        self.assertEqual(self.registry["literature_mapping_v2"]["gpu_executions"], 0)
        for pair in [("K13", "K15"), ("K16", "K19"), ("K20", "K23"), ("K24", "K27")]:
            result = trace(self.registry, *pair)
            self.assertTrue(result["edges"])
            self.assertIn("not proof", result["warning"])

    def test_quant_code_spec_nodes_and_hif4_coupling_are_indexed(self):
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["Q01"]["evidence_type"], "SPEC")
        self.assertEqual(nodes["Q03"]["evidence_type"], "SPEC")
        self.assertEqual(nodes["I04"]["evidence_type"], "ALG")
        self.assertEqual(nodes["Q11"]["evidence_type"], "OBS")
        self.assertEqual(nodes["I08"]["evidence_type"], "OPEN")
        self.assertTrue(any(e["from"]=="I06" and e["to"]=="I07"
                            for e in self.registry["edges"]))
        self.assertTrue(any(e["from"]=="Q32" and e["to"]=="I08"
                            for e in self.registry["edges"]))

    def test_realtime_studies_are_not_native_wan_streaming(self):
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["S13"]["evidence_type"], "LIT")
        self.assertEqual(nodes["S15"]["evidence_type"], "LIT")
        self.assertEqual(nodes["S20"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["Q33"]["evidence_type"], "LIT")
        self.assertIn("G10", self.registry["graph_roles"])

    def test_current_mechanism_crosswalk_anchors_papers_and_exact_synthesis_edges(self):
        crosswalk = load_crosswalk(self.registry)
        result = validate_crosswalk(self.registry)
        self.assertIsNotNone(crosswalk)
        self.assertEqual(result["mechanism_families"], 14)
        self.assertEqual(result["primary_papers"], 85)
        self.assertEqual(result["crossgraph_comparison_edges"], 67)
        self.assertEqual(len(self.registry["nodes"]), 171)
        self.assertEqual(len(self.registry["edges"]), 354)
        self.assertTrue(all(e["edge_type"] == "SCIENTIFIC_COMPARISON_OR_CONDITIONAL"
                            for e in crosswalk["synthesis_edges"]))
        families = {f["id"]: f for f in crosswalk["families"]}
        self.assertIn("I04", families["M02"]["home"])
        self.assertIn("L74", families["M05"]["papers"])
        self.assertIn("S20", families["M10"]["home"])

    def test_function_first_output_significance_and_native_structure_are_explicit(self):
        crosswalk = load_crosswalk(self.registry)
        self.assertEqual([f["id"] for f in crosswalk["families"][:2]], ["M13", "M14"])
        families = {f["id"]: f for f in crosswalk["families"]}
        self.assertIn("U07", families["M13"]["home"])
        self.assertIn("U10", families["M14"]["home"])
        self.assertIn("L84", families["M13"]["papers"])
        nodes = {n["id"]: n for n in self.registry["nodes"]}
        self.assertEqual(nodes["U02"]["evidence_type"], "NATIVE-OBS")
        self.assertEqual(nodes["U04"]["evidence_type"], "SRC")
        self.assertEqual(nodes["U08"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U11"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U14"]["evidence_type"], "OPEN")
        self.assertEqual(nodes["U15"]["evidence_type"], "LIT")
        self.assertTrue(any(e["from"]=="U14" and e["to"]=="I04"
                            for e in self.registry["edges"]))
        self.assertTrue(any(e["from"]=="U14" and e["to"]=="Q09"
                            for e in self.registry["edges"]))
        self.assertEqual(len(self.registry["function_first_graph_20261027"]["node_ids"]), 18)
        self.assertTrue(self.registry["function_first_graph_20261027"]["no_new_native_runs"])

    def test_unregistered_discovery_needs_curator(self):
        with self.assertRaises(SystemExit):
            inspect(self.registry, "F99")

    def test_open_proposal_is_not_experimental_evidence(self):
        proposed = inspect(self.registry, "P14")
        self.assertEqual(proposed["node"]["evidence_type"], "OPEN")
        self.assertEqual(proposed["node"]["availability"], "posthoc_hypothesis_only")
        self.assertGreaterEqual(len(proposed["node"]["evidence"]), 2)
        claim_path = Path(__file__).resolve().parents[1] / proposed["node"]["evidence"][0]
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        self.assertEqual(claim["review_status"], "PROPOSED")
        self.assertEqual(claim["review_refs"], [])
        self.assertFalse(claim["model_experiment_authorized"])
        candidate_route = trace(self.registry, "P06", "P14")
        self.assertEqual(len(candidate_route["edges"]), 1)
        self.assertTrue(candidate_route["edges"][0]["relation"].startswith("open_hypothesis"))

if __name__ == "__main__":
    unittest.main()
