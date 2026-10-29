import sys,unittest,math
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_all160_runtime_v2 as m

class Tensor:
 def __init__(self,shape,dtype='u8',device='cuda'):self.shape=shape;self.dtype=dtype;self.device=SimpleNamespace(type=device)
 @property
 def T(self):return Tensor(tuple(reversed(self.shape)))
 def contiguous(self):return self
 def numel(self):return math.prod(self.shape)
 def element_size(self):return 4 if self.dtype=='f32' else 2 if self.dtype=='bf16' else 1
 def is_contiguous(self):return True
 def data_ptr(self):return 10
 def stride(self):return tuple(range(len(self.shape)))
 def record_stream(self,_):pass

class MMWeight:
 def __init__(self,suffix):
  k,n=m.LINEARS[suffix];self.weight=Tensor((k,n));self.bias=Tensor((n,));self.has_lora_branch=False;self.has_diff=False;self.calls=0;self.load_state_dict=lambda *a:'original loader'
 def apply(self,value):self.calls+=1;return value
 def __setattr__(self,name,value):
  object.__setattr__(self,name,value)
  if name=='apply' and self.__dict__.get('fail_patch'):raise RuntimeError('synthetic failed apply patch')
class Buffer:
 def __init__(self,phases):self.compute_phases=phases;self.load_state_dict=lambda *a:'original loader'
 def __setattr__(self,name,value):
  object.__setattr__(self,name,value)
  if name=='load_state_dict' and self.__dict__.get('fail_patch'):raise RuntimeError('synthetic failed load patch')
class Ledger:
 def __init__(self):self.calls=[]
 def call(self,rid,cost,fn,*a):self.calls.append((rid,dict(cost)));return fn(*a)
class Api:
 def dense_prepare(self,x):return Tensor(x.shape),Tensor((1,)),Tensor((1,))
 def sharq_prepare(self,x,n):return Tensor((1,)),Tensor((1,)),Tensor((1,)),Tensor((1,)),Tensor((1,)),Tensor((1,))
 def output_scale(self,a,b):return Tensor((1,))
 def dense_gemm(self,*a):return Tensor((2,a[1].shape[0]))
 def sparse_gemm(self,*a):return Tensor((2,a[1].shape[0]))
 def dense_accum(self,*a):return a[-1]
 def bias_add(self,x,b):return x
class Quant:
 def __init__(self):self.calls=0
 def quantize_weight_shared_nvfp4(self,w):
  self.calls+=1;n,k=w.shape;return (Tensor((n,k//2)),Tensor((1,)),Tensor(((n//128+1)*128*(k//16),)),Tensor((1,),'f32'))
class Active:
 def __init__(self):self.mapping={};self.block=None;self.s=0
 def step(self,e):assert e==self.s//2;return self.s
 def mm_call(self,obj,x,meta,fn):return fn(obj,x)
def build():
 models=[];active=Active()
 for expert in range(2):
  buffers=[]
  for _ in range(2):
   a,b=MMWeight('ffn.0'),MMWeight('ffn.2');active.mapping[id(a)]=('block',expert,'ffn.0');active.mapping[id(b)]=('block',expert,'ffn.2');buffers.append(Buffer([None,None,SimpleNamespace(ffn_0=a,ffn_2=b)]))
  models.append(SimpleNamespace(transformer_infer=SimpleNamespace(offload_manager=SimpleNamespace(cuda_buffers=buffers))))
 pipe=SimpleNamespace(runner=SimpleNamespace(model=SimpleNamespace(model=models)));torch=SimpleNamespace(uint8='u8',float32='f32',is_tensor=lambda x:isinstance(x,Tensor),cuda=SimpleNamespace(current_stream=lambda:None));guard=SimpleNamespace(torch=torch,active=active);q=Quant();ledger=Ledger();events=[]
 runtime=m.SameMasterFFNRuntime(pipe,guard,Api(),q,ledger,lambda *a:{'actual':'buffer'},sync=lambda:None,emit=events.append)
 return pipe,active,runtime,q,ledger,events

def phase(pipe,active,runtime,mode):
 runtime.begin('run_'+mode,mode)
 for step in range(4):
  active.s=step;expert=step//2
  for block in range(40):
   active.block=(expert,block);obj=pipe.runner.model.model[expert].transformer_infer.offload_manager.cuda_buffers[block%2].compute_phases[2]
   obj.ffn_0.apply(Tensor((2,5120)));obj.ffn_2.apply(Tensor((2,13824)))
 return runtime.finish()
class Tests(unittest.TestCase):
 def setUp(self):
  self.bridge_patch=patch.object(m.stream_bridge,'author_call',side_effect=lambda torch,fn,*args,**kw:fn(*args,**kw));self.bridge_patch.start();self.addCleanup(self.bridge_patch.stop)
 def test_exact_all160_and_shared_static_weight_cache(self):
  pipe,active,runtime,q,ledger,events=build();mm=pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0].compute_phases[2].ffn_0;loader=mm.load_state_dict
  with runtime:
   dense=phase(pipe,active,runtime,'bf16');self.assertEqual(dense['actual_linears'],0);self.assertEqual(q.calls,0)
   cache=phase(pipe,active,runtime,'bf16_cache');self.assertEqual(cache['actual_linears'],320);self.assertEqual(q.calls,160)
   first=phase(pipe,active,runtime,'dense32');self.assertEqual(first['actual_linears'],320);self.assertEqual(first['cache_entries'],160);self.assertEqual(q.calls,160)
   second=phase(pipe,active,runtime,'sharq');self.assertEqual(second['actual_linears'],320);self.assertEqual(q.calls,160);self.assertIs(mm.load_state_dict,loader)
  self.assertEqual(mm.load_state_dict(), 'original loader');self.assertEqual(len([x for x in events if x['event']=='linear_python_entries']),640)
  self.assertEqual(sum(v.get('sharq_stream_bridge_calls',0) for _,v in ledger.calls),800);self.assertEqual(sum(v.get('sharq_bridge_event_records',0) for _,v in ledger.calls),1600);self.assertEqual(sum(v.get('sharq_bridge_event_waits',0) for _,v in ledger.calls),1600);self.assertEqual(sum(v.get('dense_residual_gemm',0) for _,v in ledger.calls),320);self.assertEqual(sum(v.get('sparse_gemm',0) for _,v in ledger.calls),320)
 def test_candidate_refuses_missing_cached_weights(self):
  pipe,active,runtime,q,ledger,events=build()
  with runtime:
   runtime.begin('candidate','dense32');active.s=0;active.block=(0,0);mm=pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0].compute_phases[2].ffn_0
   with self.assertRaisesRegex(ValueError,'BF16 neutral must complete'):mm.apply(Tensor((2,5120)))
  self.assertEqual(q.calls,0)
 def test_duplicate_and_incomplete_are_rejected(self):
  pipe,active,runtime,*_=build()
  with runtime:
   runtime.begin('x','bf16_cache');active.s=0;active.block=(0,0);ffn=pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0].compute_phases[2];ffn.ffn_0.apply(Tensor((2,5120)))
   with self.assertRaises(ValueError):ffn.ffn_0.apply(Tensor((2,5120)))
   with self.assertRaises(ValueError):runtime.finish()
 def test_no_unregistered_fp4_calls(self):
  pipe,active,runtime,*_=build()
  with runtime:
   runtime.mode='sharq';active.block=(0,0);mm=pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0].compute_phases[2].ffn_0
   with self.assertRaises(ValueError):mm.apply(Tensor((2,5120)))
 def test_failed_sparse_call_retains_partial_entry_ledger(self):
  pipe,active,runtime,q,ledger,events=build()
  def fail(*a):raise RuntimeError('synthetic sparse failure')
  runtime.api.sparse_gemm=fail
  with runtime:
   phase(pipe,active,runtime,'bf16_cache');runtime.begin('run','sharq');active.s=0;active.block=(0,0);mm=pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0].compute_phases[2].ffn_0
   with self.assertRaises(RuntimeError):mm.apply(Tensor((2,5120)))
  event=[e for e in events if e['event']=='linear_python_entries'][-1];self.assertEqual(event['predebited']['dense_residual_gemm'],1);self.assertEqual(event['invoked']['sparse_gemm'],1);self.assertNotIn('sparse_gemm',event['returned']);self.assertNotIn('dense_residual_gemm',event['invoked'])
 def test_cache_limit_rejects_before_weight_pack(self):
  pipe,active,runtime,q,ledger,events=build();runtime.cache_limit=1
  with runtime:
   runtime.begin('run','bf16_cache');active.s=0;active.block=(0,0);mm=pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0].compute_phases[2].ffn_0
   with self.assertRaises(ValueError):mm.apply(Tensor((2,5120)))
  self.assertEqual(q.calls,0)
 def test_prefetch_observes_real_source_and_never_skips_loader(self):
  pipe,active,runtime,q,ledger,events=build();torch=SimpleNamespace(bfloat16='bf16');buffer=pipe.runner.model.model[0].transformer_infer.offload_manager.cuda_buffers[0];ffn=buffer.compute_phases[2];dest={}
  for suffix,attr in (('ffn.0','ffn_0'),('ffn.2','ffn_2')):
   mm=getattr(ffn,attr);k,n=m.LINEARS[suffix]
   for field,shape in (('weight',(k,n)),('bias',(n,))):dest[f'blocks.0.{suffix}.{field}']=Tensor(shape,'bf16','cpu');setattr(mm,field+'_cuda_buffer',Tensor(shape,'bf16'))
  called=[];audit=m.OriginalFFNPrefetchAudit(pipe,torch,ledger,'run')
  for _ in range(162):self.assertEqual(audit.observe(buffer,dest,0,0,lambda *a:called.append(1) or 'ok'),'ok')
  result=audit.finish();self.assertEqual(len(called),162);self.assertEqual(result['original_FFN_prefetch_skips'],0);self.assertEqual(result['requested_W1_H2D_bytes'],162*(141557760+27648));self.assertEqual(result['requested_W2_H2D_bytes'],162*(141557760+10240))
 def test_cleanup_sync_preserves_primary_exception_and_restores_patches(self):
  pipe,active,runtime,*_=build();objects=[v[0] for v in runtime.objects.values()]
  with self.assertRaisesRegex(ValueError,'primary') as raised:
   with runtime:
    runtime.sync=lambda:(_ for _ in ()).throw(RuntimeError('cleanup drain failed'))
    raise ValueError('primary')
  self.assertTrue(any('cleanup drain failed' in x for x in raised.exception.__notes__))
  self.assertTrue(all('apply' not in obj.__dict__ for obj in objects))
 def test_cleanup_sync_failure_without_primary_propagates(self):
  pipe,active,runtime,*_=build()
  with self.assertRaisesRegex(RuntimeError,'cleanup drain failed'):
   with runtime:runtime.sync=lambda:(_ for _ in ()).throw(RuntimeError('cleanup drain failed'))
 def test_failed_apply_patch_rolls_back_every_object(self):
  pipe,active,runtime,*_=build();objects=[v[0] for v in runtime.objects.values()];objects[2].fail_patch=True
  with self.assertRaises(RuntimeError):runtime.__enter__()
  self.assertTrue(all('apply' not in obj.__dict__ for obj in objects));self.assertFalse(runtime.saved)
 def test_failed_prefetch_patch_restores_original_loaders(self):
  pipe,active,runtime,q,ledger,events=build();buffers=[b for model in pipe.runner.model.model for b in model.transformer_infer.offload_manager.cuda_buffers];original=[b.load_state_dict for b in buffers];buffers[2].fail_patch=True
  with self.assertRaises(RuntimeError):m.OriginalFFNPrefetchAudit(pipe,SimpleNamespace(),ledger,'r').__enter__()
  self.assertTrue(all(b.load_state_dict is fn for b,fn in zip(buffers,original)))
if __name__=='__main__':unittest.main()
