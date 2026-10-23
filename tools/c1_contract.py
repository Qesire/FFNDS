"""Finite, separately authorized C1 native information diagnostic."""
import contextlib
import fcntl
import subprocess
import sys
from pathlib import Path

from rollout_contract import ROOT, atomic, read, sha

METHOD = 'C1N0_PREW1_COMPLETE_GROUP_FINITE_RESPONSE'
STATE = 'R4_CLOSED_C1_NET_ACTION_DIAGNOSTIC_REGISTERED'
CAPS = {'full_generations': 8, 'standalone_FFN_forwards': 288,
        'standalone_normalization_calls': 302, 'training_steps': 0, 'weight_factorizations': 0,
        'official_evaluator_calls': 0}
SOURCES = [0, 2, 8, 10, 16, 20]
BLOCKS = [4, 17, 30]
ARMS = ['dense_DQ', 'UB1', 'swap_A', 'swap_B']


def select_swaps(groups):
    kept = set(map(int, groups)); omitted = set(range(216)) - kept
    pairs = sorted((a, b) for a in kept for b in omitted if abs(a - b) == 1)
    if len(pairs) < 2:
        pairs = [(min(kept), min(omitted)), (max(kept), max(omitted))]
    result = [pairs[0], pairs[-1]]
    if result[0] == result[1]:
        raise ValueError('Two distinct source-blind whole64-group probes required')
    return result


def validate_registration(registration, root):
    errors = []
    try:
        p = read(root / registration['protocol'])
        if (p['id'] != METHOD or sha(root / registration['protocol']) != registration['protocol_sha256']
                or p['caps'] != CAPS or p['sources'] != SOURCES or p['blocks'] != BLOCKS
                or p['steps'] != [0, 1, 2, 3] or p['arms'] != ARMS or p['automatic_retry'] is not False):
            errors.append('C1 exact finite scope changed')
        a = read(root / p['authorization']['path'])
        if (sha(root / p['authorization']['path']) != p['authorization']['sha256']
                or a['id'] != METHOD or a['user_reply'] != '那你跑啊！！！干嘛呢'
                or a['status'] != 'EXPLICIT_USER_EXECUTION_AUTHORITY' or a['caps'] != CAPS):
            errors.append('C1 separate direct execution authority missing')
        for section in ('source_sha256', 'input_sha256'):
            for name, digest in p[section].items():
                if sha(root / name) != digest:
                    errors.append('C1 immutable source/input changed: ' + name)
        if p['max_concurrent_GPUs'] != 1 or p['new_output_bytes_max'] != 1 << 30:
            errors.append('C1 resource scope changed')
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append('Malformed C1 registration: ' + type(error).__name__)
    return errors


def require_registration():
    r = subprocess.run([sys.executable, str(ROOT / 'tools/check_research_routes.py'), '--method-id', METHOD],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('C1 native execution not registered; no Torch/model import performed')
    registration = read(ROOT / 'research/CURRENT_DECISION.json')['registered_c1_diagnostic']
    errors = validate_registration(registration, ROOT)
    if errors:
        raise RuntimeError('; '.join(errors))
    return read(ROOT / registration['protocol'])


class Budget:
    def __init__(self, root):
        self.path = Path(root) / 'BUDGET.json'
        self.lock = (Path(root) / 'BUDGET.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.path.exists():
            raise RuntimeError('Preserve first C1 attempt; no implicit retry')
        self.state = {'id': METHOD, 'caps': CAPS, 'used': {k: 0 for k in CAPS}, 'debits': []}
        atomic(self.path, self.state)

    @contextlib.contextmanager
    def call(self, name, kind):
        if (kind not in CAPS or self.state['used'][kind] >= CAPS[kind]
                or any(r['name'] == name for r in self.state['debits'])):
            raise RuntimeError('C1 scope exhausted or duplicate attempt')
        row = {'name': name, 'kind': kind, 'status': 'DEBITED_BEFORE_EXECUTION'}
        self.state['used'][kind] += 1; self.state['debits'].append(row); atomic(self.path, self.state)
        try:
            yield
        except BaseException:
            row['status'] = 'FAILED_DEBIT_PRESERVED'; raise
        else:
            row['status'] = 'COMPLETED'
        finally:
            atomic(self.path, self.state)

    def close(self):
        self.lock.close()
