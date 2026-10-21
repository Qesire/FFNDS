"""Regression checks for stale plans and already-rejected experiment reentry."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("route_guard", ROOT / "tools/check_research_routes.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class ResearchRouteGuardTests(unittest.TestCase):
    def setUp(self):
        self.decision, self.triage, self.summary, self.execution = copy.deepcopy(guard.load_records(ROOT))

    def errors(self):
        return guard.validate_records(self.decision, self.triage, self.summary, self.execution)

    def method(self, method_id):
        return next(m for m in self.triage["methods"] if m["id"] == method_id)

    def test_saved_results_and_current_decision_agree(self):
        self.assertEqual(self.errors(), [])

    def test_stale_primary_route_is_rejected(self):
        self.method("R4_COMPENSATION_NONREDUNDANT_SUPPORT")["status"] = "PRIMARY_CANDIDATE"
        self.assertTrue(any("guarded disposition" in e for e in self.errors()))

    def test_old_closed_recipe_cannot_be_reactivated(self):
        method_id = next(k for k in self.decision["closed_method_statuses"] if "BO1" in k)
        self.method(method_id)["model_execution_eligibility"] = guard.READY
        self.assertTrue(any("Closed/deferred" in e for e in self.errors()))

    def test_unknown_and_closed_stage_aliases_fail_closed(self):
        for method_id in ("NEW_NAME_FOR_OLD_ROUTE", "R4E1", "R4E2", "R4E3"):
            with self.subTest(method_id=method_id):
                self.assertFalse(guard.method_eligibility(self.decision, self.triage, method_id)[0])

    def test_downstream_stage_cannot_skip_failed_prerequisite(self):
        self.method("R3_WHOLE_M_BANK_ROUTING")["model_execution_eligibility"] = guard.READY
        self.decision["eligible_model_method_ids"] = ["R3_WHOLE_M_BANK_ROUTING"]
        self.assertTrue(any("Failed prerequisite" in e for e in self.errors()))

    def test_historical_pause_cannot_replace_current_policy(self):
        self.triage["policy"]["execution_state"] = "PAUSED_BY_USER"
        self.assertTrue(any("Policy" in e for e in self.errors()))

    def test_saved_gate_cannot_be_promoted_without_recomputation(self):
        self.summary["CV"][0]["gate_pass"] = True
        self.assertTrue(any("Saved native gate mismatch" in e for e in self.errors()))

    def test_renamed_ready_route_requires_current_decision_reconciliation(self):
        self.triage["methods"].append({"id": "NEW_NAME", "status": "CANDIDATE", "model_execution_eligibility": guard.READY})
        self.assertTrue(any("Eligible methods" in e for e in self.errors()))

    def test_alias_collision_is_rejected(self):
        self.triage["methods"][0].setdefault("aliases", []).append("R4E1")
        self.assertTrue(any("Duplicate method ID/alias" in e for e in self.errors()))

    def test_archival_plans_do_not_override_current_state(self):
        self.triage["policy_history"].append({"execution_state": "PLANNED"})
        self.method("R4_COMPENSATION_NONREDUNDANT_SUPPORT")["historical_dispositions"].append({"status": "PRIMARY_CANDIDATE"})
        self.assertEqual(self.errors(), [])

    def test_cli_returns_refusal_for_closed_recipe(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(guard.main(["--root", str(ROOT), "--method-id", "R4E1"]), 2)

    def test_diagnostic_cannot_register_a_second_method(self):
        self.decision['eligible_model_method_ids'].append('UNREVIEWED_TRAINING')
        self.assertTrue(any('Only the fixed causal diagnostic' in e for e in self.errors()))

    def test_diagnostic_requires_frozen_source_and_protocol(self):
        for field in ('protocol_sha256',):
            self.decision['registered_diagnostic'][field] = '0' * 64
        self.assertTrue(any('protocol hash changed' in e for e in self.errors()))

    def test_diagnostic_does_not_release_old_closure(self):
        self.decision['registered_diagnostic']['preserved_model_closure'] = 'REOPEN_EVERYTHING'
        self.assertTrue(any('R4 closure must remain' in e for e in self.errors()))

    def test_objective_registration_requires_native_prior(self):
        self.decision['registered_diagnostic']['prior_diagnostic']['sha256'] = '0' * 64
        self.assertTrue(any('native prerequisite evidence' in e for e in self.errors()))


if __name__ == "__main__":
    unittest.main()
