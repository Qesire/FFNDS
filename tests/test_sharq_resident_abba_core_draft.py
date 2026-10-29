import pathlib,sys,tempfile,unittest,hashlib
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import sharq_resident_abba_core_draft as c
class T(unittest.TestCase):
 def test_five_phase_wiring_and_reference_gate(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);(root/'x').write_bytes(b'x');(root/'y').write_bytes(b'x');sha=hashlib.sha256(b'x').hexdigest();binding={'path':'y','bytes':1,'sha256':sha};old={**binding,'path':'x'};prior={arm:{k:old for k in (*c.STATES,'video')} for arm in ('bf16','dense32')}
   class RT:
    def __init__(self):self._selector=NS(observe=lambda r:None);self.cache={};self.modes=[]
    def set_next_resident(self,x):self.modes.append(x)
   rt=RT();saved=rt._selector.observe;outputs=[];video_paths=[]
   for rid in 'abcde':
    folder=root/rid;folder.mkdir();[(folder/(k+'.safetensors')).write_bytes(b'x') for k in c.STATES];(folder/'video').write_bytes(b'x');video_paths.append(folder/'video')
   def phase(**kw):
    resident=kw['arm'].startswith('resident')
    for _ in range(162):kw['emit']({'event':'block_load_returned'})
    for _ in range(324):rt._selector.observe({'weight_copy':'SKIPPED' if resident else 'ISSUED','bias_copy':'ISSUED','skipped_weight_logical_bytes':1,'requested_weight_logical_bytes':1,'requested_bias_logical_bytes':1})
    return {'mode':kw['mode'],'arm':kw['arm'],'ffn_sites':160,'linear_sites':320}
   def sf(rid):return lambda *args:None,{k:{**binding,'path':rid+'/'+k+'.safetensors'} for k in c.STATES}
   with patch.object(c,'native_phase',phase),patch.object(c,'cache_checkpoint',lambda cache:{'entries':160,'id':'stable'}):
    r=c.execute(torch=None,pipe=None,guard=None,runtime=rt,vm=None,run_ids=list('abcde'),image_path='i',prompt='p',output_paths=video_paths,hf=root,prior=prior,save_factory=sf,inspect_video=lambda rid,path:{**binding,'path':str(path),'frames':81,'fps':16,'size':[832,464]},check=lambda:None,authorize=lambda rid:None,charge_block=lambda *x:None,charge_compare=lambda n,f:f(),persist=lambda *x:None,emit=outputs.append,synchronize=lambda:None,phase_synchronize=lambda:None)
   self.assertEqual(len(r['reference_gate']['checks']),55);self.assertEqual(len(r['prefetch_coverage']['completed_phases']),5);self.assertIs(rt._selector.observe,saved);self.assertEqual(rt.modes,[False,True,True,False])
 def test_failure_restores_observer(self):
  class RT:
   def __init__(self):self._selector=NS(observe=lambda r:None);self.cache={}
  rt=RT();saved=rt._selector.observe
  with patch.object(c,'native_phase',side_effect=RuntimeError('model failure')):
   with self.assertRaises(RuntimeError):c.execute(torch=None,pipe=None,guard=None,runtime=rt,vm=None,run_ids=list('abcde'),image_path='i',prompt='p',output_paths=list('abcde'),hf='.',prior={'bf16':{},'dense32':{}},save_factory=lambda rid:(lambda *a:None,{}),inspect_video=lambda *a:None,check=lambda:None,authorize=lambda rid:None,charge_block=lambda *a:None,charge_compare=lambda n,f:f(),persist=lambda *a:None,emit=lambda x:None,synchronize=lambda:None,phase_synchronize=lambda:None)
  self.assertIs(rt._selector.observe,saved)
if __name__=='__main__':unittest.main()
