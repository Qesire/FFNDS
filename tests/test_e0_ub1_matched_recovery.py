import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_ub1_matched_recovery.py';spec=importlib.util.spec_from_file_location('recovery',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RecoveryPreflightTests(unittest.TestCase):
    def test_folds_disjoint_and_cover_all(self):
        for train,held in m.folds():
            self.assertEqual(len(train),12);self.assertEqual(len(held),12);self.assertFalse(set(train)&set(held));self.assertEqual(set(train)|set(held),set(range(24)))
    def test_budget_fail_before_extra_fit(self):
        counts={'RR_fit':31,'consumer_ridge_fit':16};m.charge(counts,'RR_fit');self.assertEqual(counts['RR_fit'],32)
        with self.assertRaises(RuntimeError):m.charge(counts,'RR_fit')
        with self.assertRaises(RuntimeError):m.charge(counts,'consumer_ridge_fit')
    def test_wrong_hash_and_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'p').write_bytes(b'x');m.verify_bindings([{'path':'p','sha256':hashlib.sha256(b'x').hexdigest()}],root)
            with self.assertRaises(ValueError):m.verify_bindings([{'path':'p','sha256':'0'*64}],root)
            with self.assertRaises(ValueError):m.verify_bindings([{'path':'../p','sha256':'0'*64}],root)
    def test_pending_or_unregistered_fails(self):
        plan={'id':m.ID,'limits':dict(m.LIMITS),'positions':[[0,0],[0,39],[3,0],[3,39]],'assets_finalized':False,'math_sha256':'a'*64}
        with self.assertRaises(ValueError):m.authorize(plan,'z',{'experiment_authority':{'current_user_authorized_scopes':[]}})

class InputIdentityTests(unittest.TestCase):
    def test_wrong_scope_rows_and_source_rejected(self):
        c={'index':0,'step':0,'block':0,'expert':'high_noise_model','sha256':'good','rows':[1,2]};src={'source_id':'id','source_group':'i2v_subject:0','seed':42}
        record={'status':'PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP','index':0,'input':{'input_provenance':{'image_sha256':'id'},'source_group':'i2v_subject:0','calibration_seed':42},'scopes':[{'step':0,'block':0,'expert':'high_noise_model','sha256':'good','rows':256,'selected_rows':[1,2]}]}
        m.validate_input_scope(c,record,src);record['scopes'][0]['selected_rows']=[2,1]
        with self.assertRaises(ValueError):m.validate_input_scope(c,record,src)
        record['scopes'][0]['selected_rows']=[1,2];record['input']['calibration_seed']=43
        with self.assertRaises(ValueError):m.validate_input_scope(c,record,src)

class FoldAccessTests(unittest.TestCase):
    def test_training_accessor_excludes_held_sources(self):
        data={i:object() for i in range(24)};train,held=m.folds()[0]
        selected=m.training_subset(data,train,held)
        self.assertEqual(selected,[data[i] for i in train]);self.assertTrue(all(data[i] not in selected for i in held))
        with self.assertRaises(ValueError):m.training_subset(data,train,train)

class MathIntegrationTests(unittest.TestCase):
    def test_heldout_mutation_does_not_change_training_support(self):
        import sys,torch
        sys.path.insert(0,str(p.parent))
        import ub1_recovery_math as api
        counts={};charge=lambda k:counts.__setitem__(k,counts.get(k,0)+1)
        torch.manual_seed(9);w=torch.randn(4,128).bfloat16();gate=torch.ones(1,4,dtype=torch.bfloat16)
        dataset={i:torch.full((2,128),float(i)/16,dtype=torch.bfloat16) for i in range(24)}
        train,held=m.folds()[0];gram=api.group_weight_gram(w,gate,charge)
        def select():return api.select_supports(torch.cat(m.training_subset(dataset,train,held)),gram,1,charge)
        before=select();dataset[held[0]].fill_(1000);after=select()
        self.assertTrue(torch.equal(before['cmean'],after['cmean']));self.assertTrue(torch.equal(before['flap'],after['flap']))
        self.assertEqual(counts,{'group_weight_gram':1,'group_activation_gram':2})

class SourcewiseForwardTests(unittest.TestCase):
    def test_training_main_and_residual_use_only_source_rows(self):
        import torch
        x=torch.arange(12*256*2,dtype=torch.float32).reshape(12*256,2);seen=[]
        def linear_one(batch):seen.append(tuple(batch.shape));return batch+1
        main=m.sourcewise_forward(x,linear_one)
        self.assertEqual(seen,[(256,2)]*12);self.assertTrue(torch.equal(main,x+1))
        seen.clear()
        def rr(batch):return linear_one(linear_one(batch))
        output=m.sourcewise_forward(x,rr)
        self.assertEqual(seen,[(256,2)]*24);self.assertTrue(torch.equal(output,x+2))
        with self.assertRaises(ValueError):m.sourcewise_forward(x[:-1],linear_one)
    def test_new_budget_exact(self):
        counts={'BF16_forward_linear':2303};m.charge(counts,'BF16_forward_linear')
        with self.assertRaises(RuntimeError):m.charge(counts,'BF16_forward_linear')
