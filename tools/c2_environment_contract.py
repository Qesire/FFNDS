"""Fixed native-environment repair; no authority inherited from closed C2N2."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from c2_contract import ARMS, CAPS, ROOT, Budget as OriginalBudget, atomic, read, sha

METHOD = 'C2N2_ENVIRONMENT_PRESERVED_NORM_SCALED_SMOKE'
STATE = 'R4_CLOSED_C2_ENVIRONMENT_SMOKE_REGISTERED'
STATUS = 'REGISTERED_FINITE_C2_ENVIRONMENT_SMOKE'
KEY = 'registered_c2_environment_smoke'


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
                or p['CUDA_visibility_binding'] != 'NVML_UUID'
                or p['preserve_project_model_PYTHONPATH'] is not True
                or p['native_platform_check_enabled'] is not True):
            errors.append('C2N2 exact engineering scope changed')
        a = read(root / p['authorization']['path'])
        if (sha(root / p['authorization']['path']) != p['authorization']['sha256']
                or a['status'] != 'EXPLICIT_USER_EXECUTION_AUTHORITY' or a['id'] != METHOD
                or a['caps'] != CAPS or not a['user_reply'].strip()):
            errors.append('C2N2 separate retry authority missing')
        parent = read(root / p['parent_result']['path'])
        if (sha(root / p['parent_result']['path']) != p['parent_result']['sha256']
                or parent['id'] != 'C2N1_UUID_RESERVED_NORM_SCALED_SMOKE'
                or parent['completed_videos'] != 0 or any(parent['actual_budget'].values())):
            errors.append('C2N2 parent is not the preserved zero-forward startup failure')
        for field in ('source_sha256', 'input_sha256'):
            for name, expected in p[field].items():
                if sha(root / name) != expected: errors.append('C2N2 source/input changed: ' + name)
        required = {'tools/startup_gpu_reservation.py', 'tools/run_c2_environment_smoke.py',
                    'tools/c2_environment_contract.py', 'scripts/worker_wait_uuid_idle_exec.sh',
                    'tools/check_research_routes.py', 'scripts/worker_c2_model_env.sh',
                    'tools/check_c2_launch_environment.py'}
        if not required.issubset(p['source_sha256']): errors.append('C2N2 startup source bindings incomplete')
        original = read(root / 'research/20261023/C2N0_registered_protocol_v1.json')
        parent_protocol = read(root / 'research/CURRENT_DECISION.json')['registered_c2_reserved_smoke']['protocol']
        if sha(root / parent_protocol) != parent['protocol_sha256']:
            errors.append('C2N2 historical parent protocol changed')
        proof = read(root / p['environment_CPU_preflight']['path'])
        if (sha(root / p['environment_CPU_preflight']['path']) != p['environment_CPU_preflight']['sha256']
                or proof['status'] != 'PASS_NATIVE_IMPORTS_ASSETS_AND_ALL160_METADATA_CPU_ONLY'
                or proof['CUDA_initialized'] is not False or proof['metadata_sites'] != 160
                or proof['new_GPU_FFN_norm_selector_generation_calls'] != 0):
            errors.append('C2N2 exact import/asset/metadata CPU proof missing')
        if p['input_sha256'] != original['input_sha256']:
            errors.append('C2N2 input/data scope differs from parent')
        for name in ('tools/norm_scaled_row_dispatch.py', 'tools/ub1_row_dispatch.py', 'tools/row_dispatch_kernels.py'):
            if p['source_sha256'].get(name) != original['source_sha256'][name]:
                errors.append('C2N2 changed the frozen scientific operator: ' + name)
    except (OSError, ValueError, KeyError, TypeError):
        errors.append('C2N2 registered protocol or separate authority missing/malformed')
    return errors


def verify_model_environment():
    """Catch the observed path-clearing bug before Torch import or CUDA."""
    if os.environ.get('SKIP_PLATFORM_CHECK', 'False') in ('1', 'True'):
        raise RuntimeError('Native C2 must retain the real platform availability check')
    expected = {
        'HiFloat4': ROOT / '.runtime/code/HiF4_Sp/HiFloat4/__init__.py',
        'lightx2v': ROOT / '.runtime/code/LightX2V/lightx2v/__init__.py',
    }
    for name, path in expected.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != path.resolve():
            raise RuntimeError('Native model import path missing or changed before CUDA: ' + name)


def require_registration():
    result = subprocess.run([sys.executable, str(ROOT / 'tools/check_research_routes.py'), '--method-id', METHOD],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('C2N2 engineering retry is not separately authorized; Torch/CUDA not imported')
    registration = read(ROOT / 'research/CURRENT_DECISION.json')[KEY]
    errors = validate_registration(registration, ROOT)
    if errors: raise RuntimeError('; '.join(errors))
    return read(ROOT / registration['protocol'])


class Budget(OriginalBudget):
    def __init__(self, folder):
        super().__init__(folder)
        self.state['id'] = METHOD
        atomic(self.path, self.state)
