"""No-model guard and finite budget for the pending C2 paired smoke."""
import contextlib
import fcntl
import subprocess
import sys
from pathlib import Path
from rollout_contract import ROOT, atomic, read, sha

METHOD = 'C2N0_NORM_SCALED_ENERGY_PAIRED_SMOKE'
STATE = 'R4_CLOSED_C2_NORM_SCALED_SMOKE_REGISTERED'
CAPS = {'full_generations': 6, 'standalone_FFN_forwards': 0,
        'standalone_normalization_calls': 2, 'standalone_selector_calls': 3,
        'training_steps': 0, 'weight_factorizations': 0, 'official_evaluator_calls': 0}
ARMS = ['control_observed_UB1', 'dense_DQ', 'UB1', 'fixed_stride', 'dynamic_residual', 'norm_scaled_energy']


def validate_registration(registration, root):
    errors = []
    try:
        p = read(root / registration['protocol'])
        if (p['id'] != METHOD or p['caps'] != CAPS or p['arms'] != ARMS or p['source_index'] != 6
                or p['status'] != 'REGISTERED_FINITE_C2_NATIVE_SMOKE'
                or sha(root / registration['protocol']) != registration['protocol_sha256']
                or p['automatic_retry'] is not False or p['reset_between_steps'] is not False
                or p['max_concurrent_GPUs'] != 1 or p['new_output_bytes_max'] != 1 << 29):
            errors.append('C2 exact finite scope changed')
        a = read(root / p['authorization']['path'])
        if (sha(root / p['authorization']['path']) != p['authorization']['sha256']
                or a['id'] != METHOD or a['status'] != 'EXPLICIT_USER_EXECUTION_AUTHORITY'
                or a['caps'] != CAPS or not a['user_reply'].strip()):
            errors.append('C2 separate execution authority missing')
        for section in ('source_sha256', 'input_sha256'):
            for name, expected in p[section].items():
                if sha(root / name) != expected: errors.append('C2 immutable source/input changed: ' + name)
    except (OSError, ValueError, KeyError, TypeError):
        errors.append('C2 registration missing or malformed')
    return errors


def require_registration():
    result = subprocess.run([sys.executable, str(ROOT / 'tools/check_research_routes.py'), '--method-id', METHOD],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('C2 is not separately registered/authorized; Torch/model/CUDA not imported')
    r = read(ROOT / 'research/CURRENT_DECISION.json')['registered_c2_smoke']
    errors = validate_registration(r, ROOT)
    if errors: raise RuntimeError('; '.join(errors))
    return read(ROOT / r['protocol'])


class Budget:
    def __init__(self, folder):
        self.path = Path(folder) / 'BUDGET.json'
        if self.path.exists(): raise RuntimeError('Preserve first C2 attempt; no retry')
        self.lock = (Path(folder) / 'BUDGET.lock').open('a')
        try: fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close(); raise
        if self.path.exists():
            self.lock.close(); raise RuntimeError('Preserve first C2 attempt; no retry')
        self.state = {'id': METHOD, 'caps': CAPS, 'used': {k: 0 for k in CAPS}, 'debits': []}
        atomic(self.path, self.state)

    @contextlib.contextmanager
    def call(self, name, kind):
        if (kind not in CAPS or self.state['used'][kind] >= CAPS[kind]
                or any(r['name'] == name for r in self.state['debits'])):
            raise RuntimeError('C2 duplicate attempt or scope exhausted')
        row = {'name': name, 'kind': kind, 'status': 'DEBITED_BEFORE_EXECUTION'}
        self.state['used'][kind] += 1; self.state['debits'].append(row); atomic(self.path, self.state)
        try: yield
        except BaseException:
            row['status'] = 'FAILED_DEBIT_PRESERVED'; raise
        else: row['status'] = 'COMPLETED'
        finally: atomic(self.path, self.state)

    def close(self): self.lock.close()
