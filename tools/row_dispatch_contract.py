"""Finite registration/debit contract for a full-160-site row-dispatch test."""
import contextlib
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from rollout_contract import ROOT, atomic, read, sha

METHOD = 'R16V0_FULL160_UB1_DENSE_ROW_DISPATCH'
STATE = 'R4_CLOSED_FULL160_ROW_DISPATCH_REGISTERED'
CAPS = {'full_generations': 5, 'standalone_FFN_forwards': 164,
        'weight_factorizations': 0, 'training_steps': 0}
ARMS = ['dense_DQ', 'UB1', 'fixed_stride', 'fixed_hash', 'dynamic_residual']


def validate_registration(registration, root):
    errors = []
    try:
        path = root / registration['protocol']
        p = read(path)
        if (sha(path) != registration['protocol_sha256'] or p['id'] != METHOD or
                p['status'] != 'REGISTERED_FIXED_FULL160_ROW_DISPATCH' or
                p['caps'] != CAPS or p['arms'] != ARMS or p['positions'] != [[s, b] for s in range(4) for b in range(40)] or
                p['benchmark_positions'] != [[s, 20] for s in range(4)] or
                p['benchmark'] != {'control_repeats': 1, 'warmups': 2, 'timed_repeats': 5,
                                   'UB1_reference_controls': 4} or
                p['source_index'] != 0 or p['seed'] != 42 or p['dense_row_fraction'] != .25 or
                p['expected_grid'] != [21, 29, 52] or p['max_GPUs'] != 1 or
                p['automatic_followup'] is not False or p['reset_between_steps'] is not False):
            errors.append('Full160 row-dispatch scope or immutable protocol differs')
        binding = p['authorization']
        auth = read(root / binding['path'])
        if (sha(root / binding['path']) != binding['sha256'] or auth['method_id'] != METHOD or
                auth['caps'] != CAPS or auth['status'] != 'EXPLICIT_EXACT_SCOPE_USER_AUTHORIZATION' or
                not auth['user_reply'].strip()):
            errors.append('Separate scoped direct user authority is missing')
        draft = p['approved_draft']
        if (sha(root / draft['path']) != draft['sha256'] or
                auth['approved_draft_sha256'] != draft['sha256']):
            errors.append('Approved finite-scope draft changed')
        for field in ('source_sha256', 'input_sha256'):
            for relative, expected in p[field].items():
                target = root / relative
                if not target.resolve().is_relative_to(root.resolve()) or sha(target) != expected:
                    errors.append('Frozen row-dispatch binding changed: ' + relative)
        if (p['resources'] != {'new_output_bytes_max': 1 << 29, 'free_disk_reserve_bytes': 80 << 30,
                              'peak_allocated_bytes_max': 79 << 30} or
                p['official_quality_models'] != 0):
            errors.append('Row-dispatch resource/evaluator scope changed')
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append('Malformed row-dispatch registration: ' + type(error).__name__)
    return errors


def require_registration():
    result = subprocess.run([sys.executable, str(ROOT / 'tools/check_research_routes.py'),
                             '--method-id', METHOD], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Row-dispatch model execution is not registered; no torch/model import performed')
    reg = read(ROOT / 'research/CURRENT_DECISION.json')['registered_row_dispatch']
    errors = validate_registration(reg, ROOT)
    if errors:
        raise RuntimeError('; '.join(errors))
    return read(ROOT / reg['protocol'])


class Budget:
    def __init__(self, root):
        self.path = Path(root) / 'BUDGET.json'
        self.lock = (Path(root) / 'BUDGET.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.path.exists():
            raise RuntimeError('Preserve prior row-dispatch attempt')
        self.state = {'method_id': METHOD, 'caps': CAPS, 'used': {k: 0 for k in CAPS}, 'debits': []}
        atomic(self.path, self.state)

    @contextlib.contextmanager
    def call(self, name, kind):
        if kind not in ('full_generations', 'standalone_FFN_forwards'):
            raise RuntimeError('No training/factorization is allowed')
        if any(row['name'] == name for row in self.state['debits']):
            raise RuntimeError('No implicit retry of a debited operation')
        if self.state['used'][kind] >= CAPS[kind]:
            raise RuntimeError('Row-dispatch model budget exhausted')
        self.state['used'][kind] += 1
        row = {'name': name, 'kind': kind, 'status': 'DEBITED_BEFORE_EXECUTION'}
        self.state['debits'].append(row)
        atomic(self.path, self.state)
        try:
            yield
        except BaseException:
            row['status'] = 'FAILED_DEBIT_PRESERVED'
            raise
        else:
            row['status'] = 'COMPLETED'
        finally:
            atomic(self.path, self.state)

    def close(self):
        self.lock.close()
