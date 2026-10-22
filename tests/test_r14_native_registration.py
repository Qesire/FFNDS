"""Separate R14N0 authority and finite scope cannot be inferred from R14 timing."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import check_research_routes as guard


class StripeRegistrationTests(unittest.TestCase):
    def setUp(self):
        before = json.loads((ROOT / 'research/20261022/R14N0_before_execution.json').read_text())
        self.records = list(copy.deepcopy(guard.load_records(ROOT)))
        d, t, _, _ = self.records
        d['registered_diagnostic'] = copy.deepcopy(before['registration'])
        d['execution_state'] = t['policy']['execution_state'] = guard.STRIPE_STATE
        d['eligible_model_method_ids'] = [guard.STRIPE_ID]
        d['closed_method_statuses'].pop(guard.STRIPE_ID, None)
        for method in t['methods']:
            if method.get('model_execution_eligibility') == guard.READY:
                method['model_execution_eligibility'] = 'NOT_SELECTED_IN_UNIT_FIXTURE'
            if method['id'] == guard.STRIPE_ID:
                method.update(status='REGISTERED_FIXED_NATIVE_STRIPE_VALUE_PENDING_CONTROLS',
                              model_execution_eligibility=guard.READY)
        self.protocol_path = ROOT / before['registration']['protocol']
        self.protocol = json.loads(self.protocol_path.read_text())
        self.preflight_path = ROOT / self.protocol['CPU_preflight']['path']
        self.preflight = json.loads(self.preflight_path.read_text())
        self.auth_path = ROOT / self.protocol['authorization']['path']
        self.auth = json.loads(self.auth_path.read_text())
        # Native installs are intentionally absent in clean CI. These are
        # explicit synthetic in-memory source bytes, never production proofs.
        synthetic = {}
        bindings = d['registered_diagnostic']['source_sha256']
        for relative in bindings:
            if relative.startswith('.runtime/'):
                content = ('UNIT_TEST_NATIVE_BINDING_ONLY:' + relative).encode()
                synthetic[ROOT / relative] = content
                bindings[relative] = hashlib.sha256(content).hexdigest()
        self.protocol['source_sha256'] = copy.deepcopy(bindings)
        self.preflight['source_sha256'] = copy.deepcopy(bindings)
        original_bytes, original_text = Path.read_bytes, Path.read_text

        def read_text(path, *args, **kwargs):
            for target, value in ((self.protocol_path, self.protocol), (self.preflight_path, self.preflight),
                                  (self.auth_path, self.auth)):
                if path == target:
                    return json.dumps(value)
            return original_text(path, *args, **kwargs)

        for patch in (mock.patch.object(Path, 'read_bytes', lambda p: synthetic[p] if p in synthetic else original_bytes(p)),
                      mock.patch.object(Path, 'read_text', read_text)):
            patch.start()
            self.addCleanup(patch.stop)
        self.assertEqual(self.errors(), [])

    def errors(self):
        return guard.validate_records(*self.records)

    def test_authorization_is_separate_from_backend_success(self):
        self.auth['status'] = 'SYNTHETIC_BACKEND_SUCCESS_ONLY'
        self.assertTrue(any('separate authority' in error for error in self.errors()))

    def test_model_or_linear_cap_cannot_increase(self):
        self.protocol['caps']['model_generations'] = 5
        self.assertTrue(any('controls/positions/budget' in error for error in self.errors()))

    def test_source_roles_and_point_count_cannot_change(self):
        self.protocol['inputs']['training_indices'] = [0, 1]
        self.protocol['local_two_point_count'] = 33
        self.assertTrue(any('controls/positions/budget' in error for error in self.errors()))

    def test_cpu_preparation_cannot_claim_native_execution(self):
        self.preflight['CUDA_initialized'] = True
        self.assertTrue(any('non-model prerequisites' in error for error in self.errors()))

    def test_no_automatic_rollout_or_second_method(self):
        self.protocol['automatic_followup_authorization'] = True
        self.records[0]['eligible_model_method_ids'].append('ANOTHER_MODEL')
        errors = self.errors()
        self.assertTrue(any('Only the fixed causal diagnostic' in error for error in errors))
        self.assertTrue(any('controls/positions/budget' in error for error in errors))

    def test_frozen_preparation_and_registration_sources_must_agree(self):
        self.preflight['source_sha256']['tools/stripe_native_operator.py'] = '0' * 64
        self.assertTrue(any('source/preparation/registration' in error for error in self.errors()))


if __name__ == '__main__':
    unittest.main()
