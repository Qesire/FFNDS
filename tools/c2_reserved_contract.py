"""Separately authorized engineering retry, with original C2 science unchanged."""
import subprocess
import sys

from c2_contract import ARMS, CAPS, ROOT, Budget as OriginalBudget, atomic, read, sha

METHOD = 'C2N1_UUID_RESERVED_NORM_SCALED_SMOKE'
STATE = 'R4_CLOSED_C2_UUID_RESERVED_SMOKE_REGISTERED'
STATUS = 'REGISTERED_FINITE_C2_UUID_RESERVED_SMOKE'
KEY = 'registered_c2_reserved_smoke'


def validate_registration(registration, root):
    errors = []
    try:
        p = read(root / registration['protocol'])
        if (sha(root / registration['protocol']) != registration['protocol_sha256']
                or p['id'] != METHOD or p['status'] != STATUS or p['caps'] != CAPS
                or p['arms'] != ARMS or p['source_index'] != 6
                or p['automatic_retry'] is not False or p['reset_between_steps'] is not False
                or p['max_concurrent_GPUs'] != 1 or p['new_output_bytes_max'] != 1 << 29
                or p['early_same_process_CUDA_reservation'] is not True
                or p['CUDA_visibility_binding'] != 'NVML_UUID'):
            errors.append('C2N1 exact engineering scope changed')
        a = read(root / p['authorization']['path'])
        if (sha(root / p['authorization']['path']) != p['authorization']['sha256']
                or a['status'] != 'EXPLICIT_USER_EXECUTION_AUTHORITY' or a['id'] != METHOD
                or a['caps'] != CAPS or not a['user_reply'].strip()):
            errors.append('C2N1 separate retry authority missing')
        parent = read(root / p['parent_result']['path'])
        if (sha(root / p['parent_result']['path']) != p['parent_result']['sha256']
                or parent['completed_videos'] != 0 or any(parent['actual_budget'].values())):
            errors.append('C2N1 parent is not the preserved zero-forward startup failure')
        for field in ('source_sha256', 'input_sha256'):
            for name, expected in p[field].items():
                if sha(root / name) != expected: errors.append('C2N1 source/input changed: ' + name)
        required = {'tools/startup_gpu_reservation.py', 'tools/run_c2_reserved_smoke.py',
                    'tools/c2_reserved_contract.py', 'scripts/worker_wait_uuid_idle_exec.sh',
                    'tools/check_research_routes.py'}
        if not required.issubset(p['source_sha256']): errors.append('C2N1 startup source bindings incomplete')
        original = read(root / 'research/20261023/C2N0_registered_protocol_v1.json')
        if sha(root / 'research/20261023/C2N0_registered_protocol_v1.json') != parent['protocol_sha256']:
            errors.append('C2N1 historical parent protocol changed')
        if p['input_sha256'] != original['input_sha256']:
            errors.append('C2N1 input/data scope differs from parent')
        for name in ('tools/norm_scaled_row_dispatch.py', 'tools/ub1_row_dispatch.py', 'tools/row_dispatch_kernels.py'):
            if p['source_sha256'].get(name) != original['source_sha256'][name]:
                errors.append('C2N1 changed the frozen scientific operator: ' + name)
    except (OSError, ValueError, KeyError, TypeError):
        errors.append('C2N1 registered protocol or separate authority missing/malformed')
    return errors


def require_registration():
    result = subprocess.run([sys.executable, str(ROOT / 'tools/check_research_routes.py'), '--method-id', METHOD],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('C2N1 engineering retry is not separately authorized; Torch/CUDA not imported')
    registration = read(ROOT / 'research/CURRENT_DECISION.json')[KEY]
    errors = validate_registration(registration, ROOT)
    if errors: raise RuntimeError('; '.join(errors))
    return read(ROOT / registration['protocol'])


class Budget(OriginalBudget):
    def __init__(self, folder):
        super().__init__(folder)
        self.state['id'] = METHOD
        atomic(self.path, self.state)
