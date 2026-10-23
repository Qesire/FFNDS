"""Separate finite authority for fresh quality and repeated system timing."""
import contextlib
import fcntl
from pathlib import Path
import subprocess
import sys

from rollout_contract import ROOT, atomic, read, sha

METHOD = 'R17E0_FRESH23_NINE_METRICS_AND_SYSTEM_TIMING'
STATE = 'R4_CLOSED_FRESH23_QUALITY_SYSTEM_REGISTERED'
ARMS = ['dense_DQ', 'UB1', 'fixed_stride', 'fixed_hash', 'dynamic_residual']
CAPS = {'full_generations': 145, 'metric_evaluator_calls': 45,
        'standalone_FFN_forwards': 0, 'training_steps': 0, 'weight_factorizations': 0}
CORE = ['i2v_subject', 'i2v_background', 'camera_motion', 'subject_consistency',
        'background_consistency', 'motion_smoothness', 'dynamic_degree',
        'aesthetic_quality', 'imaging_quality']


def generation_jobs():
    jobs = []
    for arm in ARMS:
        jobs.append({'name': f'warm_{arm}', 'phase': 'warmup', 'arm': arm, 'source': -1})
    for repeat in range(5):
        order = ARMS[repeat:] + ARMS[:repeat]
        for arm in order:
            jobs.append({'name': f'timing{repeat}_{arm}', 'phase': 'timing', 'arm': arm,
                         'source': -1, 'repeat': repeat})
    for source in range(23):
        other = ARMS[1:]
        offset = source % 4
        order = ['dense_DQ', *other[offset:], *other[:offset]]
        for arm in order:
            jobs.append({'name': f'quality{source:02d}_{arm}', 'phase': 'quality',
                         'arm': arm, 'source': source})
    return jobs


def validate_registration(registration, root):
    errors = []
    try:
        path = root / registration['protocol']
        p = read(path)
        if (p['id'] != METHOD or p['status'] != 'REGISTERED_FIXED_FRESH23_QUALITY_SYSTEM' or
                sha(path) != registration['protocol_sha256'] or p['caps'] != CAPS or
                p['arms'] != ARMS or p['core_dimensions'] != CORE or
                p['generation_jobs'] != generation_jobs() or p['max_concurrent_GPUs'] != 1 or
                p['new_training_or_parameter_selection'] is not False or p['automatic_followup'] is not False):
            errors.append('Fresh23 fixed generation/evaluator scope changed')
        auth = read(root / p['authorization']['path'])
        if (sha(root / p['authorization']['path']) != p['authorization']['sha256'] or
                auth['status'] != 'EXPLICIT_EXACT_SCOPE_USER_AUTHORIZATION' or
                auth['method_id'] != METHOD or auth['caps'] != CAPS or not auth['user_reply'].strip()):
            errors.append('Separate direct fresh-quality model authority missing')
        for section in ('source_sha256', 'input_sha256'):
            for name, expected in p[section].items():
                target = root / name
                if not target.resolve().is_relative_to(root.resolve()) or sha(target) != expected:
                    errors.append('Fresh23 immutable code/asset/input binding changed: ' + name)
        if (p['resources'] != {'new_output_bytes_max': 4 << 30, 'free_disk_reserve_bytes': 80 << 30,
                              'peak_allocated_bytes_max': 79 << 30} or
                p['quality_source_count'] != 23 or p['performance_source'] != 'old_calibration24_source0'):
            errors.append('Fresh23 resource/source scope changed')
        if registration.get('phase') not in ('GENERATIONS', 'METRICS'):
            errors.append('Only an explicitly bound generation or metric phase is eligible')
        if registration.get('phase') == 'METRICS':
            leaf = registration.get('metric_phase', {})
            mp_path = root / leaf['protocol']
            mp = read(mp_path)
            if (sha(mp_path) != leaf['protocol_sha256'] or mp['parent_method_id'] != METHOD or
                    mp['parent_protocol_sha256'] != registration['protocol_sha256'] or
                    mp['evaluator_call_count'] != 45 or mp['dimensions'] != CORE or mp['arms'] != ARMS or
                    sha(root / '.runtime/R17_metric_assets.json') != mp['asset_manifest_sha256']):
                errors.append('Official evaluator phase assets/source/counts were not separately frozen')
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append('Malformed fresh-quality registration: ' + type(error).__name__)
    return errors


def require_registration(phase):
    result = subprocess.run([sys.executable, str(ROOT / 'tools/check_research_routes.py'),
                             '--method-id', METHOD], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Fresh quality/system model job not authorized; no torch/model/CUDA import performed')
    registration = read(ROOT / 'research/CURRENT_DECISION.json')['registered_fresh_quality']
    if registration.get('phase') != phase:
        raise RuntimeError('This model phase is not currently eligible')
    errors = validate_registration(registration, ROOT)
    if errors:
        raise RuntimeError('; '.join(errors))
    return read(ROOT / registration['protocol'])


class Budget:
    def __init__(self, root, *, existing=False):
        self.path = Path(root) / 'BUDGET.json'
        self.lock = (Path(root) / 'BUDGET.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if existing:
            self.state = read(self.path)
            if self.state['method_id'] != METHOD or self.state['caps'] != CAPS:
                raise RuntimeError('Metric phase belongs to another generation budget')
        else:
            if self.path.exists():
                raise RuntimeError('No implicit generator retry')
            self.state = {'method_id': METHOD, 'caps': CAPS, 'used': {k: 0 for k in CAPS}, 'debits': []}
            atomic(self.path, self.state)

    @contextlib.contextmanager
    def call(self, name, kind):
        if kind not in ('full_generations', 'metric_evaluator_calls'):
            raise RuntimeError('No standalone model forwards, training or new factorization allowed')
        if (self.state['used'][kind] >= CAPS[kind] or
                any(row['name'] == name for row in self.state['debits'])):
            raise RuntimeError('Scope exhausted or operation already attempted')
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
