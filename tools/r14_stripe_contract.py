"""Fail-closed registration and persistent resource ledger for R14N0.

Pure stdlib: importing this module, checking a draft or refusing an unregistered
job must not load torch, models, CUDA or original tensor assets.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'R14N0_COMPLETE_STRIPE_NATIVE_VALUE'
ORDER = (1, 11, 0, 10)
CAPS = {'model_generations': 4, 'standalone_FFN_forwards': 32, 'extra_linear_calls': 104}
CALLS = {'teacher_replay': 2, 'UB1_replay': 4, 'complement_A': 3, 'complement_B': 3,
         'fixed': 3, 'dynamic': 3, 'oracle': 3, 'repeat_dynamic': 3}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def require_registration():
    """Eligibility, protocol, human authority and all sources precede imports."""
    check = subprocess.run([sys.executable, str(ROOT / 'tools/check_research_routes.py'),
                            '--method-id', METHOD], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError('R14N0 has no eligible registered model route; no model or CUDA imported')
    current = read(ROOT / 'research/CURRENT_DECISION.json')
    registration = current.get('registered_diagnostic', {})
    if registration.get('method_id') != METHOD:
        raise RuntimeError('R14N0 is not the selected fixed diagnostic')
    path = ROOT / registration['protocol']
    protocol = read(path)
    if sha(path) != registration['protocol_sha256'] or protocol['id'] != METHOD:
        raise RuntimeError('Immutable registration changed')
    authorization = protocol['authorization']
    auth_path = ROOT / authorization['path']
    auth = read(auth_path)
    if (sha(auth_path) != authorization['sha256'] or auth['status'] != 'EXPLICIT_USER_AUTHORIZATION'
            or auth['authorized_method_id'] != METHOD):
        raise RuntimeError('Separate direct human authority required for R14N0')
    if protocol['status'] != 'REGISTERED_FIXED_COMPLETE_STRIPE_NATIVE_VALUE':
        raise RuntimeError('Draft protocol cannot execute')
    if (protocol['positions'] != [[2, 20]] or protocol['inputs']['indices'] != list(ORDER)
            or protocol['caps'] != CAPS or protocol['automatic_followup_authorization'] is not False):
        raise RuntimeError('Frozen source/position/resource scope changed')
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Registered source changed: ' + name)
    draft = read(ROOT / protocol['draft']['path'])
    if sha(ROOT / protocol['draft']['path']) != protocol['draft']['sha256']:
        raise RuntimeError('Original value-study draft changed')
    if (draft['sources']['fixed_mask_training'] != [1, 11]
            or draft['sources']['nonfit_exposed_development'] != [0, 10]):
        raise RuntimeError('Original source exposure roles changed')
    return protocol, draft


class Budget:
    """Debit before work; failed calls and retries cannot silently vanish."""
    def __init__(self, root, *, create=False):
        self.root = Path(root)
        self.lock = (self.root / 'BUDGET.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.root / 'BUDGET.json'
        try:
            if create:
                write_new(self.path, {'method_id': METHOD, 'caps': CAPS, 'debits': [],
                                      'used': {key: 0 for key in CAPS}})
            self.state = read(self.path)
            if self.state['method_id'] != METHOD or self.state['caps'] != CAPS:
                raise RuntimeError('Budget belongs to another fixed study')
            expected = {key: sum(row['charge'].get(key, 0) for row in self.state['debits']) for key in CAPS}
            if self.state['used'] != expected or any(expected[key] > CAPS[key] for key in CAPS):
                raise RuntimeError('Budget journal inconsistent')
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def save(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state, indent=2) + '\n')
        temp.replace(self.path)

    def debit(self, index, label):
        if index not in ORDER:
            raise ValueError('Unregistered source')
        if label == 'capture':
            charge = {'model_generations': 1, 'extra_linear_calls': 2}
        elif label in CALLS:
            charge = {'standalone_FFN_forwards': 1, 'extra_linear_calls': CALLS[label]}
        else:
            raise ValueError('Unregistered action')
        if any(d['index'] == index and d['label'] == label for d in self.state['debits']):
            raise RuntimeError('Already debited; implicit retry forbidden')
        next_used = {key: self.state['used'][key] + charge.get(key, 0) for key in CAPS}
        if any(next_used[key] > CAPS[key] for key in CAPS):
            raise RuntimeError('Fixed resource cap exceeded')
        row = {'index': index, 'label': label, 'charge': charge, 'status': 'DEBITED_BEFORE_EXECUTION'}
        self.state['debits'].append(row)
        self.state['used'] = next_used
        self.save()
        return row

    @contextlib.contextmanager
    def call(self, index, label):
        row = self.debit(index, label)
        try:
            yield row
        except BaseException as error:
            row.update(status='FAILED_DEBIT_PRESERVED', error_type=type(error).__name__)
            self.save()
            raise
        else:
            row['status'] = 'COMPLETED'
            self.save()


def execution_root(protocol):
    """One canonical cache output root, independent of arbitrary CLI paths."""
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    relative = Path(protocol['HF_derived_run_relative'])
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Only the registered relative HF output locator is supported')
    return hf / 'ffnds-derived' / relative
