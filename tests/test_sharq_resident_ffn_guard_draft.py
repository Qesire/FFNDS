import pathlib,sys,unittest
from types import SimpleNamespace as NS
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_ffn_guard_draft import require_load_context,load_source_guard,require_compute_context
class T(unittest.TestCase):
 def test_context_rejects_wrong_stream_or_phase(self):
  mm=object();other=object();phase0=NS(ffn_0=other,ffn_2=object());phase1=NS(ffn_0=mm,ffn_2=object());mgr=NS(lazy_load=False,cuda_buffers=[NS(compute_phases=[None,None,phase0]),NS(compute_phases=[None,None,phase1])],cuda_load_stream='L',init_stream='I');t=NS(cuda=NS(current_stream=lambda:'L'));r=NS(mode='dense32',rid='x',cache={(0,0,'ffn.0'):(1,2,3,4)})
  require_load_context(t,mgr,mm,(0,0,'ffn.0'),r)
  with self.assertRaises(ValueError):require_load_context(t,mgr,other,(0,0,'ffn.0'),r)
  t.cuda.current_stream=lambda:'I';require_load_context(t,mgr,other,(0,0,'ffn.0'),r)
  t.cuda.current_stream=lambda:'bad'
  with self.assertRaises(ValueError):require_load_context(t,mgr,mm,(0,0,'ffn.0'),r)
  t.cuda.current_stream=lambda:'L';r.mode='bf16'
  with self.assertRaises(ValueError):require_load_context(t,mgr,mm,(0,0,'ffn.0'),r)
 def test_compute_stream_and_active_buffer(self):
  mm=object();other=object();phase=NS(ffn_0=mm,ffn_2=other);manager=NS(lazy_load=False,cuda_buffers=[NS(compute_phases=[None,None,phase]),NS(compute_phases=[None,None,NS(ffn_0=other,ffn_2=mm)])],compute_stream='C');torch=NS(cuda=NS(current_stream=lambda:'C'))
  require_compute_context(torch,manager,mm,'ffn.0')
  with self.assertRaises(ValueError):require_compute_context(torch,manager,other,'ffn.0')
  torch.cuda.current_stream=lambda:'L'
  with self.assertRaises(ValueError):require_compute_context(torch,manager,mm,'ffn.0')
 def test_source_dtype_shape_and_device(self):
  def tensor(shape,device,dtype='bf16'):return NS(shape=shape,device=NS(type=device),dtype=dtype)
  t=NS(bfloat16='bf16',is_tensor=lambda x:hasattr(x,'shape'));mm=NS(weight_cuda_buffer=tensor((4,6),'cuda'),bias_cuda_buffer=tensor((6,),'cuda'));w=tensor((4,6),'cpu');b=tensor((6,),'cpu');load_source_guard(t,mm,(0,0,'ffn.0'),w,b,{'ffn.0':(4,6)})
  with self.assertRaises(ValueError):load_source_guard(t,mm,(0,0,'ffn.0'),w,tensor((6,),'cpu','bad'),{'ffn.0':(4,6)})
if __name__=='__main__':unittest.main()
