import hashlib,json,tempfile,unittest
from pathlib import Path
from tools import audit_compact6016_partial_bank as m

def write(root,name,obj):
 p=root/name;p.write_text(json.dumps(obj));return {'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
class Tests(unittest.TestCase):
 def cell(self,root):
  pack=root/'pack';pack.write_bytes(b'abc');ex={'path':'pack','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest(),'reloaded_all_tensor_bytes_equal':True};rows=[{'index':i,'step':s,'role':'TRAIN' if i<16 else 'HELDOUT_EXPOSED_DIAGNOSTIC','actual_BF16_gate_relative_SSE':i+s} for i in range(24) for s in (0,1)];summary=[{'index':i,'value':i+.5} for i in range(24)];v={'run_id':'x','expert':'high_noise_model','block':0,'initialization':'energy94','status':'COMPLETE_256_UPDATES_AND_INITIAL_FINAL_48_SOURCE_STEP_DIAGNOSTICS','steps':[0,1],'GPU_zero_LoRA_and_native_STE_byte_checks':dict.fromkeys(('post_GELU','pre_Q2','post_Q2','pre_gate_output'),True),'GPU_gate_gradient_norms':{'lora_A1':0.,'lora_B1':1.,'lora_A2':0.,'lora_B2':2.},'zero_merge_base_weight_bitwise':[False,False],'export':ex,'epoch_source_step_loss_means':[0]*8,'diagnostics':{'initial':rows,'final_folded_BF16':rows},'source_cluster_means':{k:{'actual_BF16_gate_relative_SSE':summary} for k in ('initial','final_folded_BF16')}};entry={k:v[k] for k in ('run_id','expert','block','initialization','status','export')};entry['result']=write(root,'result',v);return entry
 def test_cell_pair_mean_and_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);c=self.cell(h);paid=[];r=m.audit_cell(h,c,paid.append);self.assertEqual(paid,[3]);self.assertEqual(r['diagnostic_summary']['initial']['equal_source_mean'],20.)
   (h/'pack').write_bytes(b'xyz')
   with self.assertRaises(ValueError):m.audit_cell(h,c,paid.append)
 def test_false_gate_or_zero_gradient_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);c=self.cell(h);v=json.loads((h/'result').read_text());v['GPU_zero_LoRA_and_native_STE_byte_checks']['post_Q2']=False;c['result']=write(h,'result',v)
   with self.assertRaises(ValueError):m.audit_cell(h,c,lambda n:None)
   v['GPU_zero_LoRA_and_native_STE_byte_checks']['post_Q2']=True;v['GPU_gate_gradient_norms']['lora_B1']=0.;c['result']=write(h,'result',v)
   with self.assertRaises(ValueError):m.audit_cell(h,c,lambda n:None)
 def test_scope_and_missing_pair_rejected(self):
  with self.assertRaises(ValueError):m.audit_parts(Path('/tmp'),[],'sha',lambda n:None)
  with self.assertRaises(ValueError):m.authorize({'id':m.ID,'assets_finalized':False},'sha',{})
if __name__=='__main__':unittest.main()
