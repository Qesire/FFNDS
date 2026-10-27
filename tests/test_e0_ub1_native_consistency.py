import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_ub1_native_consistency.py';spec=importlib.util.spec_from_file_location('ub1consistency',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PreflightTests(unittest.TestCase):
    def test_case_and_authorization(self):
        p={'id':m.ID,'limits':dict(m.LIMITS),'cases':[{'source':s,'step':t,'block':b} for s,t,b in sorted(m.EXPECTED)]};a={'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'ok','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
        m.authorize(p,'ok',a)
        with self.assertRaises(ValueError):m.authorize(p,'bad',a)
        p['cases'][0]=p['cases'][1]
        with self.assertRaises(ValueError):m.authorize(p,'ok',a)
    def test_rows_and_counter(self):
        m.validate_rows([1,3],5,2)
        for ids in ([1,1],[-1,2],[1,5]):
            with self.assertRaises(ValueError):m.validate_rows(ids,5,2)
        c={'native_Q2':47};m.count_call(c,'native_Q2')
        with self.assertRaises(RuntimeError):m.count_call(c,'native_Q2')
    def test_asset_tamper_fails(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'asset';p.write_bytes(b'x');m.verify_bindings([{'path':'asset','sha256':hashlib.sha256(b'x').hexdigest()}],t)
            with self.assertRaises(ValueError):m.verify_bindings([{'path':'asset','sha256':'0'*64}],t)
            with self.assertRaises(ValueError):m.verify_bindings([{'path':'../asset','sha256':'0'*64}],t)

class PrimitiveTests(unittest.TestCase):
    def test_padding_and_bit_identity(self):
        import torch
        x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);full,rows=m.padded(x,[3,1],5)
        self.assertTrue(m.bit_equal(full.index_select(0,rows),x))
        self.assertEqual(m.compare(x,x)['mismatched_elements'],0)
        zero=torch.tensor([[0.]],dtype=torch.bfloat16);neg=torch.tensor([[-0.]],dtype=torch.bfloat16)
        self.assertFalse(m.bit_equal(zero,neg))

class IdentityTests(unittest.TestCase):
    def test_scope_rejects_wrong_row_and_panel(self):
        case={'source':0,'seed':42,'step':0,'block':0,'expert':'high_noise_model','panel':'step0_block00.safetensors','panel_sha256':'good','rows':[1,2]}
        run={'source_index':0,'seed':42,'scopes':[{'step':0,'block':0,'expert':'high_noise_model','sha256':'good','path':'step0_block00.safetensors','row_plan':{'rows':[1,2]}}]}
        self.assertEqual(m.validate_scope(case,run)['rows'],[1,2])
        run['scopes'][0]['row_plan']['rows']=[2,1]
        with self.assertRaises(ValueError):m.validate_scope(case,run)
        run['scopes'][0]['row_plan']['rows']=[1,2];run['scopes'][0]['sha256']='bad'
        with self.assertRaises(ValueError):m.validate_scope(case,run)
    def test_wrong_block_coefficient_path_fails_before_read(self):
        p=Path('/tmp/step0_block39.npz');q=Path('/tmp/step0_block39.json')
        with self.assertRaisesRegex(ValueError,'position path'):m.load_coeff(p,q,p,q,0,0,'cpu','x','y')

class CompareValidationTests(unittest.TestCase):
    def test_reject_broadcast_and_wrong_dtype(self):
        import torch
        a=torch.zeros((2,2),dtype=torch.bfloat16)
        with self.assertRaises(ValueError):m.compare(a,torch.zeros((1,2),dtype=torch.bfloat16))
        with self.assertRaises(ValueError):m.compare(a,torch.zeros((2,2),dtype=torch.float32))

class CoefficientIdentityTests(unittest.TestCase):
    def test_wrong_metadata_block_rejected_even_when_filename_matches(self):
        import json,numpy as np
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);n=root/'step0_block00.npz';j=root/'step0_block00.json'
            np.savez(n,groups=np.arange(54,dtype=np.int64),b2=np.zeros(5120,dtype=np.float32))
            j.write_text(json.dumps({'status':'COMPLETE','payload_sha256':m.sha(n),'data':{'step':0,'block':39}}))
            with self.assertRaisesRegex(ValueError,'position differs'):m.load_coeff(n,j,n,j,0,0,'cpu','w','b')
