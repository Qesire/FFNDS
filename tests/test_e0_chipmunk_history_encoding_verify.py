import unittest
from tools.e0_chipmunk_history_encoding_verify import validate_typed
class TypedVerifyTest(unittest.TestCase):
 def test_rejects_inflight_and_nan(self):
  row={'gated_relative_l2':0.,'absolute_max':0.,'relative_l2_vs_A':0.,'relative_l2_vs_dense':0.,'absmax_vs_dense':0.};a={'predebited':{'x':1},'completed':{'x':1},'inflight':None,'metrics':{'A_B_bitwise_equal':True,'masks_equal':True,'teacher_read_after_all_actions':True,'dense_vs_saved_teacher':{'relative_l2':0.,'absmax':0.},'arms':{k:dict(row) for k in 'ABCD'}},'mask_sha256':{k:'same' for k in 'ABCD'},'operand_versions_before':{},'operand_versions_after':{},'timings':{**{'dense_'+str(i):{} for i in range(5)},**{k+'_'+str(i):{} for k in 'ABCD' for i in range(5)}},'encoding_metadata':{'fp8':{'payload_bytes':31668*13824,'shadow_bytes':64*13824*2,'scale_bytes':0,'saturated_elements':0},'int8':{'payload_bytes':31668*13824,'shadow_bytes':64*13824*2,'scale_bytes':13824*4}},'peak_GPU_allocated_bytes':1,'peak_GPU_reserved_bytes':2};validate_typed(a,{'x':1});a['inflight']={'x':'UNKNOWN'}
  with self.assertRaises(ValueError):validate_typed(a,{'x':1})
  a['inflight']=None;a['metrics']['arms']['C']['absolute_max']=float('nan')
  with self.assertRaises(ValueError):validate_typed(a,{'x':1})
