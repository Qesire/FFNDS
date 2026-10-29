import unittest
from tools.e0_chipmunk_input_state_verify import validate_typed
class TypedVerifyTest(unittest.TestCase):
 def test_rejects_inflight_and_nan(self):
  row={'gated_relative_l2':0.,'absolute_max':0.,'relative_l2_vs_A':0.,'relative_l2_vs_dense':0.,'absmax_vs_dense':0.};a={'predebited':{'x':1},'completed':{'x':1},'inflight':None,'metrics':{'same_P64_masks':True,'old_recompute':{**{k:{'relative_l2_vs_saved_h':0.,'absmax_vs_saved_h':0.} for k in ('C','D')},**{k:{'relative_l2':0.,'absmax':0.} for k in ('C_vs_D','C_vs_D_gated')}},'masks_equal':True,'teacher_read_after_all_actions':True,'dense_vs_saved_teacher':{'relative_l2':0.,'absmax':0.},'arms':{k:dict(row) for k in 'ABCD'}},'mask_sha256':{k:'same' for k in 'ABCD'},'operand_versions_before':{},'operand_versions_after':{},'timings':{**{'dense_'+str(i):{} for i in range(5)},**{k+'_'+str(i):{} for k in 'ABCD' for i in range(5)}},'encoding_metadata':{k:{'payload_bytes':31668*(13824 if k=='B' else 5120*2),'shadow_bytes':64*13824*2,'saturated_elements':0} for k in 'ABCD'},'peak_GPU_allocated_bytes':1,'peak_GPU_reserved_bytes':2};validate_typed(a,{'x':1});a['inflight']={'x':'UNKNOWN'}
  with self.assertRaises(ValueError):validate_typed(a,{'x':1})
  a['inflight']=None;a['metrics']['arms']['C']['absolute_max']=float('nan')
  with self.assertRaises(ValueError):validate_typed(a,{'x':1})
