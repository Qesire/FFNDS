"""Finite full-video execution boundaries; no model/torch/CUDA imports."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from rollout_contract import Budget,CAPS,METHOD,validate_registration


class PairedRolloutContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ffnds-pair-')
        self.root=Path(self.temp.name)
        self.p=json.loads((ROOT/'research/20261023/R15V0_paired_rollout_DRAFT.json').read_text())
        self.p['status']='REGISTERED_PAIRED_FULL_FREE_ROLLOUT_SMOKE'
        (self.root/'source.py').write_text('synthetic fixture, not an executable model\n')
        self.p['source_sha256']={'source.py':self.digest(self.root/'source.py')}
        self.p['input_sha256']={}
        self.auth={'status':'EXPLICIT_EXACT_SCOPE_USER_AUTHORIZATION','method_id':METHOD,
                   'caps':CAPS,'user_reply':'synthetic authorization fixture only'}
        self.save_auth()

    def tearDown(self):self.temp.cleanup()

    def digest(self,p):return hashlib.sha256(p.read_bytes()).hexdigest()

    def save_auth(self):
        f=self.root/'authority.json';f.write_text(json.dumps(self.auth))
        self.p['authorization']={'path':f.name,'sha256':self.digest(f)}

    def validate(self):
        f=self.root/'protocol.json';f.write_text(json.dumps(self.p))
        r={'protocol':f.name,'protocol_sha256':self.digest(f)}
        return validate_registration(r,self.root)

    def test_exact_fixture_and_no_torch_import(self):
        self.assertEqual(self.validate(),[])
        self.assertNotIn('torch',sys.modules)

    def test_missing_explicit_reply_or_changed_scope_refuses(self):
        self.auth['user_reply']='';self.save_auth();self.assertTrue(self.validate())
        self.auth['user_reply']='synthetic fixture';self.save_auth()
        self.p['positions'].append([2,21]);self.assertTrue(self.validate())

    def test_source_mutation_refuses(self):
        (self.root/'source.py').write_text('changed fixture\n')
        self.assertTrue(self.validate())

    def test_dense_step_reset_cannot_qualify_as_free_rollout(self):
        self.p['reset_between_steps']=True
        self.assertTrue(self.validate())

    def test_failed_generation_keeps_debit_and_duplicate_refuses(self):
        b=Budget(self.root)
        try:
            with self.assertRaisesRegex(RuntimeError,'fixture failure'):
                with b.call('dense_DQ'):raise RuntimeError('fixture failure')
            self.assertEqual(b.state['used']['full_generations'],1)
            self.assertEqual(b.state['debits'][0]['status'],'FAILED_DEBIT_PRESERVED')
            with self.assertRaises(RuntimeError):
                with b.call('dense_DQ'):pass
        finally:b.close()


if __name__=='__main__':unittest.main()
