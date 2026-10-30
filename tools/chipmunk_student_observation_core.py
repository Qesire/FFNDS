"""Full-M shadow observer component; requires separately registered owner.

Explicit writer and ledger are injected. Nothing returned by this observer is
used as model output. The real post-process callback must be supplied and
independently source-audited before any scientific execution.
"""
from chipmunk_student_observation_coordinates import coordinates
from chipmunk_student_observation_contract import SNAPSHOT_BLOCKS
from chipmunk_student_observation_math import paired_errors,prepare_omission,omission_energy,partition_sums

class Observer:
 def __init__(self,*,write_formula,sink,selector_indices,chunk_rows=256):
  if chunk_rows<=0:raise ValueError('chunk size')
  self.write_formula=write_formula;self.sink=sink;self.selector_indices=list(selector_indices);self.eval=coordinates(self.selector_indices);self.chunk_rows=chunk_rows
 def observe_dense(self,*,torch,ledger,rid,block,x,w1_native,b1,w2_native,b2,actual_h,actual_y,old_h,old_y,gate,residual_input,attention_input,expected_stream):
  import torch.nn.functional as F
  if torch.cuda.current_stream()!=expected_stream:raise ValueError('current stream differs')
  def charged(cost,fn):
   def wrapped():
    value=fn();torch.cuda.synchronize();return value
   return ledger.call(rid,cost,wrapped)
  h=charged({'dense_shadow_W1':1,'dense_shadow_GELU':1},lambda:F.gelu(torch.addmm(b1,x,w1_native,out=torch.empty((x.shape[0],w1_native.shape[1]),dtype=x.dtype,device=x.device)),approximate='tanh'))
  y=charged({'dense_shadow_W2':1},lambda:torch.addmm(b2,h,w2_native,out=torch.empty((x.shape[0],w2_native.shape[1]),dtype=x.dtype,device=x.device)))
  def bits(a,b):return bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
  def check():
   if not bits(h,actual_h) or not bits(y,actual_y):raise ValueError('dense shadow bytes differ')
  charged({'dense_h_Y_bitwise_checks':1},check)
  charged({'dense_metadata_write':1,'dense_gate_D2H':1,'dense_gate_D2H_bytes':20480},lambda:self.sink.dense_metadata(block,self.eval,self.selector_indices,gate))
  charged({'dense_sample_write':1,'dense_rows_H2D_calls':1,'dense_rows_H2D_bytes':2560,'dense_sample_tensor_D2H':5,'dense_raw_tensor_sha':5,'dense_sample_file_sha':1,'dense_header_reads':1,'dense_D2H_bytes':27525120,'dense_raw_sha_bytes':27525120},lambda:self.sink.dense_sample(block,self.selector_indices,self.eval,x,old_h,actual_h,old_y,actual_y))
  if block in SNAPSHOT_BLOCKS:charged({'dense_snapshot_blocks':1,'dense_snapshot_tensor_D2H':5,'dense_raw_tensor_sha':5,'dense_snapshot_file_sha':5,'dense_header_reads':5,'dense_D2H_bytes':2723954688,'dense_raw_sha_bytes':2723954688},lambda:self.sink.dense_snapshot(block,x,old_h,actual_h,old_y,actual_y))
  return charged({'postprocess_shadow_write':1},lambda:self.write_formula(actual_y,gate,0,x.shape[0],residual_input,attention_input))
 def check_postprocess(self,*,torch,ledger,rid,expected,actual):
  def check():
   ok=bool(torch.equal(expected.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8)));torch.cuda.synchronize()
   if not ok:raise ValueError('actual postprocess bytes differ')
  return ledger.call(rid,{'postprocess_actual_bitwise_checks':1},check)
 def observe(self,*,torch,ledger,rid,expert,block,x,w1,b1,w2,b2,w1_native,w2_native,old_h,old_y,candidate_y,mask,score,selector_rows,gate,residual_input,attention_input,expected_stream):
  import torch.nn.functional as F
  if expert!=0 or not 0<=block<40 or tuple(x.shape)!=(31668,5120) or tuple(old_h.shape)!=(31668,13824):raise ValueError('student observation geometry')
  if torch.cuda.current_stream()!=expected_stream:raise ValueError('current stream differs')
  def charged(cost,fn):
   def wrapped():
    value=fn();torch.cuda.synchronize();return value
   return ledger.call(rid,cost,wrapped)
  charged({'selector_identity':1,'selector_identity_D2H':1,'selector_identity_D2H_bytes':512},lambda: (_ for _ in ()).throw(ValueError('selector differs')) if selector_rows.detach().cpu().tolist()!=self.selector_indices else None)
  # One full current W1 and W2, explicitly separate from real action.
  current_h=charged({'full_shadow_W1':1,'full_shadow_GELU':1},lambda:F.gelu(torch.addmm(b1,x,w1_native,out=torch.empty((x.shape[0],w1_native.shape[1]),dtype=x.dtype,device=x.device)),approximate='tanh'))
  dense_y=charged({'full_shadow_W2':1},lambda:torch.addmm(b2,current_h,w2_native,out=torch.empty((x.shape[0],w2_native.shape[1]),dtype=x.dtype,device=x.device)))
  # Persist exact actual indices, scores and coordinates, never feed them back.
  charged({'mask_index_body_copies':1,'mask_index_D2H_bytes':40960,'selector_score_copies':1,'selector_score_D2H_bytes':55296,'gate_copies':1,'gate_D2H_bytes':20480,'metadata_selector_D2H_calls':1,'metadata_selector_D2H_bytes':512,'coordinate_rows_64_and_256':1,'metadata_json_write':1},lambda:self.sink.metadata(block,mask,score,selector_rows,gate,self.eval))
  if block in SNAPSHOT_BLOCKS:
   charged({'snapshot_blocks':1,'snapshot_tensor_D2H':6,'snapshot_file_hash':6,'raw_tensor_sha':6,'typed_header_reads':6,'student_D2H_bytes':3048235008,'student_raw_sha_bytes':3048235008},lambda:self.sink.snapshot(block,x,old_h,current_h,old_y,dense_y,candidate_y,gate,mask,score,selector_rows,self.eval))
  prepared=charged({'mask_validation':1,'mask_unique':1,'mask_bounds_scalar_checks':2,'W2_FP32_and_gather':1,'diagonal_weight_reduction':1},lambda:prepare_omission(torch,w2,mask,gate))
  charged({'sample_320_gather_and_write':1,'sample_selector_D2H_calls':1,'sample_selector_D2H_bytes':512,'student_rows_H2D_calls':1,'student_rows_H2D_bytes':2560,'sample_tensor_D2H':6,'sample_file_hash':1,'raw_tensor_sha':6,'typed_header_reads':1,'student_D2H_bytes':30801920,'student_raw_sha_bytes':30801920},lambda:self.sink.sample(block,selector_rows,self.eval,x,old_h,current_h,old_y,dense_y,candidate_y))
  # Partition sums are additive across chunks, preserving all rows. Gates and
  # exact writeback semantics are supplied by the source-bound owner.
  for start in range(0,x.shape[0],self.chunk_rows):
   end=min(start+self.chunk_rows,x.shape[0])
   def measure():
    g=self.sink.gate_slice(gate,start,end)
    err=paired_errors(torch,candidate_y[start:end],dense_y[start:end],g,lambda v:self.write_formula(v,gate,start,end,residual_input,attention_input))
    ideal=omission_energy(torch,current_h[start:end].float()-old_h[start:end].float(),prepared)
    stats=partition_sums(torch,start,end,err['pre_error'],err['ideal_gated_error'],err['actual_write_error'],ideal,err['dense_pre'],err['dense_gated'],err['dense_write'])
    return self.sink.partition(block,start,end,stats)
   charged({'observer_chunk':1,'omitted_projection_matmul':1,'diagonal_gemv':1,'observer_output_gate_and_write_comparisons':1,'finite_all_scalar_checks':3,'partition_bincounts':14,'partition_stat_D2H':14,'chunk_stats_write':1},measure)
  charged({'site_stats_write':1},lambda:self.sink.finish_site(block))
  expected_write=charged({'postprocess_shadow_write':1},lambda:self.write_formula(candidate_y,gate,0,x.shape[0],residual_input,attention_input))
  charged({'observer_sync_before_release':1},lambda:None)
  return expected_write
