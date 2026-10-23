"""Synthetic no-model registration checks for the pending startup repair."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from c2_reserved_contract import ARMS, CAPS, METHOD, STATUS, validate_registration
from c2_contract import atomic, sha


class ReservedContract(unittest.TestCase):
    def fixture(self, root):
        math_sources = ['tools/norm_scaled_row_dispatch.py','tools/ub1_row_dispatch.py','tools/row_dispatch_kernels.py']
        sources = math_sources + ['tools/startup_gpu_reservation.py','tools/run_c2_reserved_smoke.py',
                                  'tools/c2_reserved_contract.py','scripts/worker_wait_uuid_idle_exec.sh',
                                  'tools/check_research_routes.py']
        for name in sources:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic CPU registration fixture\n')
        atomic(root/'input.json', {'synthetic':True})
        inputs={'input.json':sha(root/'input.json')}
        original={'source_sha256':{n:sha(root/n) for n in math_sources},'input_sha256':inputs}
        original_path=root/'research/20261023/C2N0_registered_protocol_v1.json';original_path.parent.mkdir(parents=True)
        atomic(original_path,original)
        atomic(root/'parent.json', {'completed_videos':0,'actual_budget':{k:0 for k in CAPS},'protocol_sha256':sha(original_path)})
        atomic(root/'auth.json', {'id':METHOD,'status':'EXPLICIT_USER_EXECUTION_AUTHORITY','caps':CAPS,'user_reply':'synthetic fixture only'})
        p={'id':METHOD,'status':STATUS,'caps':CAPS,'arms':ARMS,'source_index':6,'automatic_retry':False,
           'reset_between_steps':False,'max_concurrent_GPUs':1,'new_output_bytes_max':1<<29,
           'early_same_process_CUDA_reservation':True,'CUDA_visibility_binding':'NVML_UUID',
           'authorization':{'path':'auth.json','sha256':sha(root/'auth.json')},
           'parent_result':{'path':'parent.json','sha256':sha(root/'parent.json')},
           'source_sha256':{n:sha(root/n) for n in sources},'input_sha256':inputs}
        return p

    def validate(self, root, p):
        atomic(root/'protocol.json',p)
        return validate_registration({'protocol':'protocol.json','protocol_sha256':sha(root/'protocol.json')},root)

    def test_exact_synthetic_scope_and_separate_authority_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root);self.assertEqual(self.validate(root,p),[])
            p['authorization']=None
            self.assertTrue(self.validate(root,p))

    def test_parent_with_any_forward_exposure_cannot_be_called_zero_forward_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root)
            parent=json.loads((root/'parent.json').read_text());parent['actual_budget']['full_generations']=1
            atomic(root/'parent.json',parent);p['parent_result']['sha256']=sha(root/'parent.json')
            self.assertTrue(any('zero-forward' in s for s in self.validate(root,p)))

    def test_changing_operator_under_engineering_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root);name='tools/norm_scaled_row_dispatch.py'
            (root/name).write_text('changed scientific operator\n');p['source_sha256'][name]=sha(root/name)
            self.assertTrue(any('scientific operator' in s for s in self.validate(root,p)))

    def test_scope_or_startup_source_omission_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root);p['max_concurrent_GPUs']=2
            self.assertTrue(any('scope changed' in s for s in self.validate(root,p)))
            p['max_concurrent_GPUs']=1;p['source_sha256'].pop('tools/startup_gpu_reservation.py')
            self.assertTrue(any('bindings incomplete' in s for s in self.validate(root,p)))


if __name__ == '__main__': unittest.main()
