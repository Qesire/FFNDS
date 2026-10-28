"""Pure-stdlib validation of the current FFNDS status/dependency task ledger.

No model, network, GPU, timer or GitHub workflow execution.
"""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "research/agent_protocol/CURRENT_TASK_LEDGER_20261028.json"


class CurrentTaskLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(LEDGER.read_text(encoding="utf-8"))
        cls.tasks = {t["id"]: t for t in cls.data["tasks"]}

    def test_unique_ids_statuses_and_nonempty_next_actions(self):
        tasks = self.data["tasks"]
        self.assertEqual(len(tasks), len(self.tasks))
        self.assertIn("P0-B0-PREP", self.tasks)
        self.assertIn("P0-B0-NATIVE", self.tasks)
        self.assertIn("P0-B1-FINITE", self.tasks)
        self.assertIn("P1-Q0-ASSET-AUDIT", self.tasks)
        self.assertEqual(self.tasks["P0-B0-PREP"]["status"], "READY_STATIC")
        self.assertEqual(self.tasks["P0-B0-NATIVE"]["status"], "OPEN_NEEDS_REGISTERED_NATIVE")
        self.assertEqual(self.tasks["P0-B1-FINITE"]["status"], "WAITING_ON_EVIDENCE")
        self.assertEqual(self.tasks["P1-A0-BASELINE"]["status"], "PARTIAL_NATIVE")
        for task in tasks:
            self.assertIn(task["status"], self.data["status_legend"])
            self.assertTrue(task["next_action"])
            self.assertTrue(task["output"])
            self.assertTrue(task["basis"])
            for source in task["basis"]:
                self.assertTrue((ROOT / source).is_file(), f"Missing basis: {task['id']} -> {source}")

    def test_dependency_dag_and_primary_route(self):
        deps = {k: v["dependencies"] for k, v in self.tasks.items()}
        visited, visiting = set(), set()

        def visit(name):
            self.assertIn(name, self.tasks)
            if name in visited:
                return
            self.assertNotIn(name, visiting, f"Cycle detected at {name}")
            visiting.add(name)
            for nxt in deps[name]:
                visit(nxt)
            visiting.remove(name)
            visited.add(name)

        for task_id in self.tasks:
            visit(task_id)
        self.assertIn("P0-B0-PREP", deps["P0-B0-NATIVE"])
        self.assertIn("P0-B0-NATIVE", deps["P0-B1-FINITE"])
        self.assertIn("P0-B1-FINITE", deps["P1-B2-STUDENT"])
        self.assertEqual(self.tasks["P4-REALTIME"]["status"], "DEFERRED")

    def test_authority_and_navigation_are_not_duplicated(self):
        auth = self.data["authority_and_scope"]
        for key in ("scientific_goal", "method_authority", "execution_authority",
                    "execution_handoff", "scientific_discussion", "navigation"):
            self.assertTrue((ROOT / auth[key]).is_file())
        self.assertEqual(auth["execution_authority"], "research/ACTIVE_RESEARCH.json")
        self.assertTrue(auth["independent_review_not_peer_vote"])
        self.assertEqual(auth["last_verified_pending_cpu_gpu_model_jobs"], 0)
        self.assertIn("CURRENT_TASK_LEDGER_20261028.json",
                      (ROOT / "research/RESEARCH_NAVIGATION.md").read_text(encoding="utf-8"))

    def test_b0_and_qcfb_have_no_fabricated_native_completion(self):
        self.assertNotEqual(self.tasks["P0-B0-NATIVE"]["status"], "COMPLETE_HISTORICAL")
        self.assertEqual(self.tasks["P3-Q3-NATIVE-AND-VIDEO"]["status"],
                         "WAITING_ON_EVIDENCE")
        self.assertIn("NO_NEW_STUDENT_MODEL_JOB", self.tasks["P1-B2-STUDENT"]["execution"])


if __name__ == "__main__":
    unittest.main()
