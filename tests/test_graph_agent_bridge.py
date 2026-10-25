"""Pure-static tests of G0–G5 graph ↔ Agent node/evidence index."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from graph_agent_bridge import load, validate, inspect, trace


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
