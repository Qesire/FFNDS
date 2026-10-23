"""The environment repair must not inherit the failed attempt's authority."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from c2_environment_contract import ARMS, CAPS, METHOD, STATUS, validate_registration
from c2_contract import sha


class EnvironmentAuthority(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ['tools/startup_gpu_reservation.py', 'tools/run_c2_environment_smoke.py',
            'tools/c2_environment_contract.py', 'scripts/worker_wait_uuid_idle_exec.sh',
            'tools/check_research_routes.py', 'scripts/worker_c2_model_env.sh',
            'tools/check_c2_launch_environment.py', 'tools/norm_scaled_row_dispatch.py',
            'tools/ub1_row_dispatch.py', 'tools/row_dispatch_kernels.py']
        for n in self.names:
            p = self.root / n; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('bound source\n')
        sources = {n: sha(self.root / n) for n in self.names}
        self.write('n1_protocol.json', {'id': 'historical'})
        parent = {'id': 'C2N1_UUID_RESERVED_NORM_SCALED_SMOKE', 'completed_videos': 0,
                  'actual_budget': {k: 0 for k in CAPS}, 'protocol_sha256': sha(self.root / 'n1_protocol.json')}
        self.write('parent.json', parent)
        self.write('research/CURRENT_DECISION.json', {'registered_c2_reserved_smoke': {'protocol': 'n1_protocol.json'}})
        self.write('research/20261023/C2N0_registered_protocol_v1.json', {'input_sha256': {}, 'source_sha256': sources})
        self.write('authority.json', {'id': METHOD, 'status': 'EXPLICIT_USER_EXECUTION_AUTHORITY',
                                    'caps': CAPS, 'user_reply': 'explicit fresh authority'})
        self.write('proof.json', {'status': 'PASS_NATIVE_IMPORTS_ASSETS_AND_ALL160_METADATA_CPU_ONLY',
                                'CUDA_initialized': False, 'metadata_sites': 160,
                                'new_GPU_FFN_norm_selector_generation_calls': 0})
        self.protocol = {'id': METHOD, 'status': STATUS, 'caps': CAPS, 'arms': ARMS,
            'source_index': 6, 'automatic_retry': False, 'reset_between_steps': False,
            'max_concurrent_GPUs': 1, 'new_output_bytes_max': 1 << 29,
            'early_same_process_CUDA_reservation': True, 'CUDA_visibility_binding': 'NVML_UUID',
            'preserve_project_model_PYTHONPATH': True, 'native_platform_check_enabled': True,
            'authorization': {'path': 'authority.json', 'sha256': sha(self.root / 'authority.json')},
            'parent_result': {'path': 'parent.json', 'sha256': sha(self.root / 'parent.json')},
            'source_sha256': sources, 'input_sha256': {},
            'environment_CPU_preflight': {'path': 'proof.json', 'sha256': sha(self.root / 'proof.json')}}

    def write(self, name, value):
        p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))

    def check(self, p=None):
        self.write('protocol.json', self.protocol if p is None else p)
        return validate_registration({'protocol': 'protocol.json', 'protocol_sha256': sha(self.root / 'protocol.json')}, self.root)

    def test_exact_scope_with_fresh_authority_accepts(self):
        self.assertEqual(self.check(), [])

    def test_previous_attempt_authority_rejected(self):
        authority = json.loads((self.root / 'authority.json').read_text())
        authority['id'] = 'C2N1_UUID_RESERVED_NORM_SCALED_SMOKE'; self.write('authority.json', authority)
        self.protocol['authorization']['sha256'] = sha(self.root / 'authority.json')
        self.assertTrue(self.check())

    def test_native_platform_skip_not_allowed(self):
        p = copy.deepcopy(self.protocol); p['native_platform_check_enabled'] = False
        self.assertTrue(self.check(p))

    def test_cpu_proof_cannot_claim_cuda_execution(self):
        proof = json.loads((self.root / 'proof.json').read_text()); proof['CUDA_initialized'] = True
        self.write('proof.json', proof); self.protocol['environment_CPU_preflight']['sha256'] = sha(self.root / 'proof.json')
        self.assertTrue(self.check())

    def test_nonzero_parent_scope_not_reclassified_as_startup(self):
        parent = json.loads((self.root / 'parent.json').read_text()); parent['completed_videos'] = 1
        self.write('parent.json', parent); self.protocol['parent_result']['sha256'] = sha(self.root / 'parent.json')
        self.assertTrue(self.check())


if __name__ == '__main__':
    unittest.main()
