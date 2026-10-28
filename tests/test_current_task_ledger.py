"""Pure-stdlib validation of the current FFNDS status/dependency task ledger.

No model, network, GPU, timer or GitHub workflow execution.
"""
import json
import hashlib
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
        self.assertEqual(self.tasks["P0-B0-PREP"]["status"], "COMPLETE_NATIVE")
        self.assertEqual(self.tasks["P0-B0-NATIVE"]["status"], "COMPLETE_NATIVE")
        self.assertEqual(self.tasks["P0-B1-FINITE"]["status"], "PARTIAL_NATIVE")
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
        self.assertGreaterEqual(auth["last_verified_pending_cpu_gpu_model_jobs"], 0)
        self.assertIsNone(auth["current_global_GPU_limit"])
        active = json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text())
        self.assertEqual(active["experiment_authority"]["active_resource_policy"],
                         "resource_scheduling_policy_20261028")
        self.assertIsNone(active["resource_scheduling_policy_20261028"]["global_gpu_card_cap"])
        self.assertEqual(auth["current_session_explicit_GPU_limit"], 2)
        self.assertEqual(active["current_session_resource_constraint_20261028"]
                         ["max_simultaneously_assigned_idle_GPUs"], 2)
        self.assertIn("CURRENT_TASK_LEDGER_20261028.json",
                      (ROOT / "research/RESEARCH_NAVIGATION.md").read_text(encoding="utf-8"))

    def test_b0_completion_requires_bound_actual_evidence_and_b1_stays_partial(self):
        binding = self.tasks["P0-B0-NATIVE"]["native_result"]
        raw = (ROOT / binding["path"]).read_bytes()
        self.assertEqual(len(raw), binding["bytes"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), binding["sha256"])
        report = json.loads(raw)
        self.assertEqual(report["b0"]["launch_exit"], 0)
        self.assertEqual(report["b0"]["FFN_calls"], 160)
        self.assertEqual(report["b0"]["historical_raw_fp32_poststate_match"], 4)
        active = json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text())
        scope = next(s for s in active["experiment_authority"]["current_user_authorized_scopes"]
                     if s["id"] == "E0_B_FINAL_BLOCK_NATIVE_CAPTURE_HEAD_ASSET_20261028")
        self.assertEqual(scope["result_sha256"], report["b0"]["run_sha256"])
        self.assertEqual(scope["actual_process_exit"]["launcher_returncode"], 0)
        self.assertTrue(scope["actual_process_exit"]["process_exited"])
        self.assertIn("general_conditional_paired_functions_beyond_this_finite_pool",
                      self.tasks["P0-B1-FINITE"]["outstanding_components"])
        latest = self.tasks["P0-B1-FINITE"]["latest_completed_result"]
        raw = (ROOT / latest["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), latest["sha256"])
        self.assertEqual(len(raw), latest["bytes"])
        result = json.loads(raw)
        self.assertEqual(result["labels"]["totals"]["sources"], 24)
        self.assertEqual(result["labels"]["totals"]["cases"], 96)
        selection = result["paired_CPU_M256"]["selection"]
        self.assertEqual({selection[k] for k in ("paired", "source_shuffled", "marginal",
                                                "PCA_in_shared_span", "shared_head_only")},
                         {"PCA_in_shared_span"})
        self.assertEqual(result["paired_CPU_M256"]["full_M_native_calls"], 0)
        self.assertFalse(result["B0_full_M_neutral"]["full_state_reproduction"])
        self.assertEqual(self.tasks["P3-Q3-NATIVE-AND-VIDEO"]["status"],
                         "WAITING_ON_EVIDENCE")
        self.assertIn("BH_DIRECT65_H128", self.tasks["P1-B2-STUDENT"]["execution"])


if __name__ == "__main__":
    unittest.main()
