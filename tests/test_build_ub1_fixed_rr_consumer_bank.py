import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/build_ub1_fixed_rr_consumer_bank.py';spec=importlib.util.spec_from_file_location('bank',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class BuilderTests(unittest.TestCase):
    def test_exact_part_coverage_and_counts(self):
        positions=[{'step':s,'block':b} for s in range(4) for b in range(40)];inputs=[{'index':i,'path':str(i)} for i in range(24)];p={'positions':positions,'inputs':inputs}
        seen=set()
        for part in range(4):
            a=m.validate_cases(p,part);self.assertEqual(len(a),40);these={(x['step'],x['block']) for x in a};self.assertFalse(seen&these);seen|=these
        self.assertEqual(len(seen),160)
        counts={'BF16_forward_linear':3839,'consumer_delta_fit':40};m.charge(counts,'BF16_forward_linear')
        with self.assertRaises(RuntimeError):m.charge(counts,'BF16_forward_linear')
        with self.assertRaises(RuntimeError):m.charge(counts,'consumer_delta_fit')
    def test_original_scope_mismatch_fails(self):
        source={'source_id':'id','source_group':'i2v_subject:0','seed':42};c={'index':0,'step':0,'block':0,'sha256':'sha','rows':[1,2],'expert':'high_noise_model'}
        rec={'index':0,'status':'PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP','input':{'input_provenance':{'image_sha256':'id'},'source_group':'i2v_subject:0','calibration_seed':42},'scopes':[{'step':0,'block':0,'sha256':'sha','selected_rows':[1,2],'rows':256,'expert':'high_noise_model'}]}
        m.validate_input_scope(c,rec,source);rec['scopes'][0]['selected_rows']=[2,1]
        with self.assertRaises(ValueError):m.validate_input_scope(c,rec,source)
    def test_binding_and_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'v';p.write_bytes(b'x');m.verify_bindings([{'path':'v','sha256':hashlib.sha256(b'x').hexdigest()}],tmp)
            with self.assertRaises(ValueError):m.verify_bindings([{'path':'v','sha256':'bad'}],tmp)
        plan={'id':m.ID,'limits_per_part':dict(m.LIMITS),'assets_finalized':False,'math_sha256':'x'*64}
        with self.assertRaises(ValueError):m.authorize(plan,'none',{'experiment_authority':{'current_user_authorized_scopes':[]}},0)

class MathBoundaryTests(unittest.TestCase):
    def test_error_uses_old_bf16_branch_addition_once(self):
        import sys,torch
        sys.path.insert(0,str(p.parent));import ub1_fixed_rr_consumer_math as api
        y=torch.tensor([[257.]],dtype=torch.bfloat16);base=torch.tensor([[256.]],dtype=torch.bfloat16);rr=torch.tensor([[1.]],dtype=torch.bfloat16)
        got=api.baseline_error(y,base,rr)
        self.assertEqual(got.dtype,torch.float32);self.assertTrue(torch.equal(got,y.float()-(base+rr).float()))

class SourcewiseExecutionTests(unittest.TestCase):
    def test_one_source_baseline_delegates_exactly_three_linears(self):
        import torch
        seen=[];x=torch.ones((256,2),dtype=torch.bfloat16);ids=torch.tensor([0,1]);w=torch.ones((2,2),dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16)
        rr={'first':w,'second':w,'bias_first':b,'bias_second':b}
        def linear(a,weight,bias):seen.append(tuple(a.shape));return torch.nn.functional.linear(a,weight,bias)
        features,main,res,combined=m.baseline_for_source(x,x,ids,1.,w,b,rr,linear)
        self.assertEqual(seen,[(256,2)]*3);self.assertTrue(torch.equal(combined,main+res));self.assertEqual(features.dtype,torch.bfloat16)

class ManifestExpansionTests(unittest.TestCase):
    def test_manifest_derived_rows_and_source_scope_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);items=[];sources=[]
            for i in range(24):
                name=f'{i}.json';src={'index':i,'source_id':f'id{i}','source_group':f'group{i}','seed':42+i};sources.append(src)
                scopes=[{'step':s,'block':b,'expert':'high_noise_model' if s<2 else 'low_noise_model','rows':256,'selected_rows':list(range(256)),'path':str(root/f'{i}-{s}-{b}.safetensors'),'sha256':'f'*64,'bytes':100} for s in range(4) for b in range(40)]
                rec={'index':i,'status':'PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP','input':{'input_provenance':{'image_sha256':f'id{i}'},'source_group':f'group{i}','calibration_seed':42+i},'scopes':scopes};path=root/name;path.write_text(json.dumps(rec));items.append({'index':i,'path':name,'sha256':m.sha(path)})
            plan={'inputs':items,'sources':sources}
            seen=set()
            for part in range(4):
                got=m.expand_inputs(plan,root,part);self.assertEqual(len(got),960);self.assertTrue(all('rows' not in c for c in got));keys={(v['index'],v['step'],v['block']) for v in got};self.assertFalse(keys&seen);seen|=keys
            self.assertEqual(len(seen),3840)
            bad=json.loads((root/'0.json').read_text());bad['scopes'][0]['selected_rows'][1]=0;(root/'0.json').write_text(json.dumps(bad));items[0]['sha256']=m.sha(root/'0.json')
            with self.assertRaises(ValueError):m.expand_inputs(plan,root,0)
            bad['scopes'][0]['selected_rows'][1]=1;bad['input']['calibration_seed']=999;(root/'0.json').write_text(json.dumps(bad));items[0]['sha256']=m.sha(root/'0.json')
            with self.assertRaises(ValueError):m.expand_inputs(plan,root,0)

class PositionBindingTests(unittest.TestCase):
    def test_valid_wrong_alias_and_swapped_sidecar_rejected(self):
        names=('main_npz','main_json','rr_npz','rr_json','shard')
        pos={name:name for name in names};pos.update({name+'_sha256':'sha-'+name for name in names})
        private={name:{'path':name,'sha256':'sha-'+name} for name in names}
        m.validate_position_bindings(pos,private)
        private['wrong']={'path':'wrong','sha256':'different-but-valid-sha'};pos['main_npz']='wrong'
        with self.assertRaises(ValueError):m.validate_position_bindings(pos,private)
        pos['main_npz']='main_npz';pos['main_json']='rr_json'
        with self.assertRaises(ValueError):m.validate_position_bindings(pos,private)
    def test_four_parts_budget_totals(self):
        self.assertEqual(4*m.LIMITS['BF16_forward_linear'],15360)
        self.assertEqual(4*m.LIMITS['consumer_delta_fit'],160)
        self.assertEqual(4*m.LIMITS['persisted_consumer_pack'],160)
