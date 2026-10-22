"""D0 authority, fixed actions and controls cannot be inferred from old E0."""
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


class D0RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.records = copy.deepcopy(guard.load_records(ROOT))
        decision, triage, _, _ = self.records
        archive = json.loads((ROOT / 'research/20261022/R12_D0_before_execution.json').read_text())
        decision['registered_diagnostic'] = copy.deepcopy(archive['registration'])
        decision['execution_state'] = triage['policy']['execution_state'] = guard.PAIRED_STATE
        decision['eligible_model_method_ids'] = [guard.PAIRED_ID]
        decision['closed_method_statuses'].pop(guard.PAIRED_ID, None)
        for method in triage['methods']:
            if method.get('model_execution_eligibility') == guard.READY:
                method['model_execution_eligibility'] = 'NOT_SELECTED_IN_UNIT_FIXTURE'
            if method['id'] == guard.PAIRED_ID:
                method.update(status='REGISTERED_FIXED_R12_D0_CONTROLS_PENDING', model_execution_eligibility=guard.READY)
        self.bindings = decision['registered_diagnostic']['source_sha256']
        synthetic = {}
        for relative in self.bindings:
            if relative.startswith('.runtime/'):
                value = ('UNIT_TEST_ONLY_NATIVE_SOURCE:' + relative).encode()
                synthetic[ROOT / relative] = value
                self.bindings[relative] = hashlib.sha256(value).hexdigest()
        original = Path.read_bytes
        patcher = mock.patch.object(Path, 'read_bytes', lambda p: synthetic[p] if p in synthetic else original(p))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertEqual(self.errors(), [])

    def errors(self):
        return guard.validate_records(*self.records)

    def mutate_protocol(self, mutate):
        path = ROOT / self.records[0]['registered_diagnostic']['protocol']
        original = Path.read_text
        protocol = json.loads(original(path))
        mutate(protocol)
        with mock.patch.object(Path, 'read_text', lambda p, *a, **kw: json.dumps(protocol) if p == path else original(p, *a, **kw)):
            return self.errors()

    def test_cannot_increase_generation_budget(self):
        errors = self.mutate_protocol(lambda p: p['execution_plan'].update(pipeline_generations_max=25))
        self.assertTrue(any('controls/positions/budget changed' in e for e in errors))

    def test_cannot_change_action_or_add_pair_correction(self):
        for field, value in [('arms', {'N0': [162, 162], 'NA': [216, 162], 'NB': [108, 216]}),
                             ('pair_correction', {'raw_biases': True, 'alpha': 1, 'R256_positions': 2,
                                                  'trainable_parameters': 0, 'fitting_updates': 0})]:
            self.assertTrue(any('controls/positions/budget changed' in e
                                for e in self.mutate_protocol(lambda p: p.update({field: value}))))

    def test_cannot_relax_gate_or_bypass_controls(self):
        for mutate in [lambda p: p['gates'].update(oracle_over_UB1_max=1.),
                       lambda p: p['execution_plan'].update(interventions_require_all_controls=False)]:
            self.assertTrue(any('controls/positions/budget changed' in e for e in self.mutate_protocol(mutate)))

    def test_explicit_authorization_cannot_be_replaced_by_agenda(self):
        protocol = json.loads((ROOT / self.records[0]['registered_diagnostic']['protocol']).read_text())
        path = ROOT / protocol['authorization']['path']
        original = Path.read_text
        auth = json.loads(original(path))
        auth['status'] = 'E0_AGENDA_ONLY'
        with mock.patch.object(Path, 'read_text', lambda p, *a, **kw: json.dumps(auth) if p == path else original(p, *a, **kw)):
            self.assertTrue(any('explicit scoped authorization' in e for e in self.errors()))

    def test_missing_native_implementation_fails_closed(self):
        path = ROOT / next(p for p in self.bindings if p.startswith('.runtime/'))
        original = Path.read_bytes
        def read(p):
            if p == path:
                raise FileNotFoundError('Native install absent in unit fixture')
            return original(p)
        with mock.patch.object(Path, 'read_bytes', read):
            self.assertTrue(any('Invalid diagnostic registration' in e for e in self.errors()))


if __name__ == '__main__':
    unittest.main()
