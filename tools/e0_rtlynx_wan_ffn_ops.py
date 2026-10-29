"""Explicit no-LoRA Wan BF16 post-GELU 2:4 port; no imports or GPU setup.

The rowwise norm and ties are a declared adaptation; author code/axis is unknown.
The caller owns authorization, native backend, ledger, and all synchronization.
"""
ARMS=('dense','top2','top2_row_norm')
EPSILON=1e-8

def expected_calls(arm):
 if arm not in ARMS:raise ValueError('invalid arm')
 costs={'W1_bf16_addmm':1,'gelu_tanh':1,'gate_multiply':1}
 if arm=='dense':costs.update(W2_bf16_addmm=1)
 elif arm=='top2':costs.update(fused_top2_pack=1,cutlass_wrapper=1,cutlass_sparse_W2=1,bias_add=1)
 else:costs.update(materialize_top2=1,row_norm_and_BF16_scale=1,fused_top2_pack=1,cutlass_wrapper=1,cutlass_sparse_W2=1,bias_add=1)
 return costs

def row_compensated(torch,h,masked):
 """Scale each full hidden row in FP32, round to BF16 before packing.

Zero rows use scale=0 (sqrt(0/(0+eps))), giving zero; NaN/Inf inputs
must be rejected by caller before invoking. Scale and output are both observed.
 """
 full=h.float();selected=masked.float();scale=(full.square().sum(dim=1,keepdim=True)/(selected.square().sum(dim=1,keepdim=True)+EPSILON)).sqrt()
 return (selected*scale).to(torch.bfloat16).contiguous(),scale

def forward(arm,x,w1,b1,w2,b2,gate,api,ledger):
 if arm not in ARMS:raise ValueError('invalid arm')
 packed=meta=None
 a=ledger.call('W1_bf16_addmm',api.addmm,b1,x,w1.T)
 h=ledger.call('gelu_tanh',api.gelu,a)
 if arm=='dense':y=ledger.call('W2_bf16_addmm',api.addmm,b2,h,w2.T);scale=None
 else:
  scale=None
  if arm=='top2_row_norm':
   masked=ledger.call('materialize_top2',api.materialize,h)
   value,scale=ledger.call('row_norm_and_BF16_scale',api.compensate,h,masked[:h.shape[0]])
  else:value=h
  packed,meta=ledger.call('fused_top2_pack',api.pack,value)
  sparse=ledger.call('cutlass_wrapper',api.wrap,packed,meta)
  raw=ledger.call('cutlass_sparse_W2',api.sparse_mm,sparse,w2.T,h.shape[0])
  y=ledger.call('bias_add',api.bias,raw,b2)
 gated=ledger.call('gate_multiply',api.mul,y,gate)
 return {'a':a,'h':h,'y':y,'gated':gated,'norm_scale':scale,'packed':packed,'metadata':meta}
