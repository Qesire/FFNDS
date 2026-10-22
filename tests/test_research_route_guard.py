"""Regression checks for stale plans and already-rejected experiment reentry."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("route_guard", ROOT / "tools/check_research_routes.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class ResearchRouteGuardTests(unittest.TestCase):
    def setUp(self):
        self.decision, self.triage, self.summary, self.execution = copy.deepcopy(guard.load_records(ROOT))
        if self.decision['execution_state'] in (guard.STRIPE_STATE, guard.ROLLOUT_STATE):
            # These legacy policy mutations concern the completed R4/D0 state.
            # R14N0 and R15V0 authority have independent registration tests.
            before = json.loads((ROOT / 'research/20261022/R14N0_before_execution.json').read_text())
            self.decision, self.triage = copy.deepcopy((before['decision_before'], before['triage_before']))
        # These tests exercise policy/registration mutations, not installed model
        # source identity. Give archived native bindings explicit synthetic bytes
        # in the IN-MEMORY fixture so the suite also runs in a clean checkout.
        # The production guard and original archived hashes remain unchanged.
        self.native_fixture_bytes = {}
        original_read_bytes = Path.read_bytes

        def read_bytes(path):
            if path in self.native_fixture_bytes:
                return self.native_fixture_bytes[path]
            return original_read_bytes(path)

        patcher = mock.patch.object(Path, 'read_bytes', read_bytes)
        patcher.start()
        self.addCleanup(patcher.stop)
        if self.decision['execution_state'] == guard.PAIRED_STATE:
            self.use_native_binding_fixture()

    def use_native_binding_fixture(self):
        bindings = self.decision['registered_diagnostic']['source_sha256']
        for relative in bindings:
            if Path(relative).parts[0] == '.runtime':
                data = ('UNIT_TEST_NATIVE_BINDING_ONLY:' + relative).encode()
                self.native_fixture_bytes[ROOT / relative] = data
                bindings[relative] = hashlib.sha256(data).hexdigest()

    def errors(self):
        return guard.validate_records(self.decision, self.triage, self.summary, self.execution)

    def method(self, method_id):
        return next(m for m in self.triage["methods"] if m["id"] == method_id)

    def use_archived_registration(self):
        """Exercise the real frozen registration independently of live selection."""
        archive = json.loads((ROOT / 'research/20261021/R8F2_before_execution.json').read_text())
        self.decision['registered_diagnostic'] = copy.deepcopy(archive['registration'])
        self.decision['execution_state'] = guard.ENCODING_STATE
        self.triage['policy']['execution_state'] = guard.ENCODING_STATE
        self.decision['eligible_model_method_ids'] = [guard.ENCODING_ID]
        for method in self.triage['methods']:
            if method.get('model_execution_eligibility') == guard.READY:
                method['model_execution_eligibility'] = 'NOT_SELECTED_IN_ARCHIVED_FIXTURE'
        self.decision['closed_method_statuses'].pop(guard.ENCODING_ID, None)
        self.method(guard.ENCODING_ID)['status'] = 'REGISTERED_FIXED_ENCODING_PENDING'
        self.method(guard.ENCODING_ID)['model_execution_eligibility'] = guard.READY
        self.assertEqual(self.errors(), [])

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
        # CLI reads fresh records, unlike the in-memory native-source fixture.
        # A clean public checkout intentionally lacks ignored native installs;
        # exercise this policy-only CLI check in its historical closed state.
        decision = copy.deepcopy(self.decision)
        triage = copy.deepcopy(self.triage)
        decision['execution_state'] = triage['policy']['execution_state'] = guard.CURRENT_STATE
        decision['eligible_model_method_ids'] = []
        for method in triage['methods']:
            if method.get('model_execution_eligibility') == guard.READY:
                method['model_execution_eligibility'] = 'NOT_SELECTED_IN_CLI_FIXTURE'
        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(guard, 'load_records', return_value=(decision, triage, self.summary, self.execution)):
                self.assertEqual(guard.main(["--root", str(ROOT), "--method-id", "R4E1"]), 2)

    def test_diagnostic_cannot_register_a_second_method(self):
        self.use_archived_registration()
        self.decision['eligible_model_method_ids'].append('UNREVIEWED_TRAINING')
        self.assertTrue(any('Only the fixed causal diagnostic' in e for e in self.errors()))

    def test_diagnostic_requires_frozen_source_and_protocol(self):
        self.use_archived_registration()
        for field in ('protocol_sha256',):
            self.decision['registered_diagnostic'][field] = '0' * 64
        self.assertTrue(any('protocol hash changed' in e for e in self.errors()))

    def test_diagnostic_does_not_release_old_closure(self):
        self.use_archived_registration()
        self.decision['registered_diagnostic']['preserved_model_closure'] = 'REOPEN_EVERYTHING'
        self.assertTrue(any('R4 closure must remain' in e for e in self.errors()))

    def test_objective_registration_requires_native_prior(self):
        self.use_archived_registration()
        self.decision['registered_diagnostic']['prior_diagnostic']['sha256'] = '0' * 64
        self.assertTrue(any('native prerequisite evidence' in e for e in self.errors()))

    def test_unrun_review_drafts_are_not_executable(self):
        for method_id in (guard.POPULATION_ID,):
            with self.subTest(method_id=method_id):
                self.assertFalse(guard.method_eligibility(self.decision, self.triage, method_id)[0])

    def use_head_registration(self):
        archive = json.loads((ROOT / 'research/20261021/R11G1_before_execution.json').read_text())
        self.decision['registered_diagnostic'] = copy.deepcopy(archive['registration'])
        self.decision['execution_state'] = self.triage['policy']['execution_state'] = guard.HEAD_STATE
        self.decision['eligible_model_method_ids'] = [guard.HEAD_ID]
        self.decision['closed_method_statuses'].pop(guard.HEAD_ID, None)
        for method in self.triage['methods']:
            if method.get('model_execution_eligibility') == guard.READY:
                method['model_execution_eligibility'] = 'NOT_SELECTED_IN_ARCHIVED_FIXTURE'
        self.method(guard.HEAD_ID).update(status='REGISTERED_FIXED_READINESS_PENDING', model_execution_eligibility=guard.READY)
        self.use_native_binding_fixture()
        self.assertEqual(self.errors(), [])

    def test_native_head_registration_does_not_release_training(self):
        self.use_head_registration()
        self.decision['eligible_model_method_ids'].append('R10_NATIVE_WINDOW_ROLLOUT_RECOVERY')
        self.assertTrue(any('Only the fixed causal diagnostic' in e for e in self.errors()))

    def test_missing_native_source_still_fails_closed(self):
        self.use_head_registration()
        missing = next(iter(self.native_fixture_bytes))
        original = Path.read_bytes

        def read(path):
            if path == missing:
                raise FileNotFoundError('Native source unavailable in unit fixture')
            return original(path)

        with mock.patch.object(Path, 'read_bytes', read):
            self.assertTrue(any('Invalid diagnostic registration' in e for e in self.errors()))

    def test_native_head_update_budget_is_checked(self):
        self.use_head_registration()
        original = Path.read_text
        protocol = ROOT / self.decision['registered_diagnostic']['protocol']
        altered = json.loads(original(protocol))
        altered['execution_plan']['gradient_updates'] = 2

        def read(path, *args, **kwargs):
            return json.dumps(altered) if path == protocol else original(path, *args, **kwargs)

        with mock.patch.object(Path, 'read_text', read):
            self.assertTrue(any('controls/positions/budget changed' in e for e in self.errors()))

    def use_matched_registration(self):
        archive = json.loads((ROOT / 'research/20261021/R11T1_before_execution.json').read_text())
        self.decision['registered_diagnostic'] = copy.deepcopy(archive['registration'])
        self.decision['execution_state'] = self.triage['policy']['execution_state'] = guard.MATCHED_STATE
        self.decision['eligible_model_method_ids'] = [guard.MATCHED_ID]
        self.decision['closed_method_statuses'].pop(guard.MATCHED_ID, None)
        for method in self.triage['methods']:
            if method.get('model_execution_eligibility') == guard.READY:
                method['model_execution_eligibility'] = 'NOT_SELECTED_IN_ARCHIVED_FIXTURE'
        self.method(guard.MATCHED_ID).update(status='REGISTERED_FIXED_MATCHED_OBJECTIVE_PENDING', model_execution_eligibility=guard.READY)
        self.use_native_binding_fixture()
        self.assertEqual(self.errors(), [])

    def test_matched_training_requires_verified_gradient_prerequisite(self):
        self.use_matched_registration()
        self.decision['registered_diagnostic']['prior_diagnostic']['sha256'] = '0' * 64
        self.assertTrue(any('exact native gradient/memory' in e for e in self.errors()))

    def test_matched_training_cannot_increase_iterations(self):
        self.use_matched_registration()
        original = Path.read_text
        protocol = ROOT / self.decision['registered_diagnostic']['protocol']
        altered = json.loads(original(protocol))
        altered['optimization']['steps_per_arm'] = 21

        def read(path, *args, **kwargs):
            return json.dumps(altered) if path == protocol else original(path, *args, **kwargs)

        with mock.patch.object(Path, 'read_text', read):
            self.assertTrue(any('controls/positions/budget changed' in e for e in self.errors()))


if __name__ == "__main__":
    unittest.main()
