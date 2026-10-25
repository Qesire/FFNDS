"""Pure-static tests of G0–G4 graph ↔ Agent node/evidence index."""
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

    def test_candidate_cost_path_is_labeled_not_runtime_proof(self):
        path = trace(self.registry, "G18", "P12")
        self.assertTrue(path["edges"])
        self.assertIn("not proof", path["warning"])
        self.assertTrue(any(e["relation"].startswith("posthoc") or
                            e["relation"].startswith("candidate")
                            for e in path["edges"]))

    def test_unregistered_discovery_needs_curator(self):
        with self.assertRaises(SystemExit):
            inspect(self.registry, "F99")


if __name__ == "__main__":
    unittest.main()
