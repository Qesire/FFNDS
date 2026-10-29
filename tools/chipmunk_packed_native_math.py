"""Packed BF16 strong control; old mask hash is a validity gate, not selector input."""
import hashlib,time
import torch
import torch.nn.functional as F
from chipmunk_fullm_proxy_math import online_mask,oracle_mask

def mask_check(mask,expected):
    actual=hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()
    if actual!=expected:raise ValueError('frozen mask SHA differs; stop before packed action')

def packed_action(x,w1,b1,w2,old_h,old_y,gate,mask,*,mm_fp32=None):
    """All gather, cache clone/update and delta operations belong in event."""
    if mm_fp32 is None:mm_fp32=lambda a,b:torch.mm(a,b,out_dtype=torch.float32)
    hc=old_h.clone();yc=old_y.clone()
    packed_w1=w1.index_select(0,mask);packed_b1=b1.index_select(0,mask);packed_w2=w2.index_select(1,mask).contiguous()
    old_selected=hc.index_select(1,mask)
    pre=F.linear(x,packed_w1,packed_b1);new=F.gelu(pre,approximate='tanh');delta=(new.float()-old_selected.float()).to(torch.bfloat16)
    hc.index_copy_(1,mask,new)
    fp32=mm_fp32(delta,packed_w2.transpose(0,1));new_y=(yc.float()+fp32).to(torch.bfloat16)
    return (new_y*gate).to(torch.bfloat16)

def execute_pair(x0,x1,w1,b1,w2,b2,gate,permutation,k,random_indices,expected,*,measured,mm_fp32=None):
    """measured(name, predebit, thunk) times only thunk, not persisted ledger."""
    def dense(x):
        pre=F.linear(x,w1,b1);h=F.gelu(pre,approximate='tanh');return pre,h,F.linear(h,w2,b2)
    old_mean,old_h,old_y=measured('old_cache',{'dense_w1':1,'gelu':1,'dense_w2':1,'old_mean':1},lambda:(lambda pre,h,y:(pre.mean(dim=0),h,y))(*dense(x0)))
    masks={};guard_wall=[]
    def check(mask,mode):
        t=time.perf_counter();mask_check(mask,expected[mode]);guard_wall.append({'mode':mode,'synchronized_host_seconds':time.perf_counter()-t})
    def evaluate(mode):
        mask=online_mask(mode,x1,w1,b1,old_h,old_mean,permutation,k,random_indices);check(mask,mode);masks[mode]=mask;return packed_action(x1,w1,b1,w2,old_h,old_y,gate,mask,mm_fp32=mm_fp32)
    base={'packed_action':1,'mask_guard':1,'gather_w1':1,'gather_w2':1,'gather_bias':1,'clone_h':1,'clone_y':1,'gather_h':1,'W1_gemm':1,'selected_gelu':1,'delta_round':1,'update_h':1,'W2_gemm_fp32':1,'accumulate_y':1,'gate_product':1}
    results={}
    for mode in ('mean','p4','p64','random'):
        results[mode]=[]
        debit={**base,'selector':1}
        if mode!='random':debit['proxy_W1']=1
        if mode in ('p4','p64'):debit['proxy_gelu']=1
        measured(f'cold_{mode}',debit,lambda mode=mode:evaluate(mode))
        for repeat in range(3):
            results[mode].append(measured(f'online_{mode}_{repeat}',debit,lambda mode=mode:evaluate(mode)))
    def current_dense():
        pre,h,y=dense(x1);return h,y,(y*gate).to(torch.bfloat16)
    measured('dense_reference_cold',{'dense_w1':1,'gelu':1,'dense_w2':1,'gate_product':1},current_dense)
    refs=[measured(f'dense_reference_{i}',{'dense_w1':1,'gelu':1,'dense_w2':1,'gate_product':1},current_dense) for i in range(3)]
    current_h,current_y,current_gated=refs[-1]
    oracle=measured('oracle_selection',{'oracle_selection':1},lambda:oracle_mask(current_h,old_h,k))
    def oracle_action():
        check(oracle,'oracle');masks['oracle']=oracle;return packed_action(x1,w1,b1,w2,old_h,old_y,gate,oracle,mm_fp32=mm_fp32)
    results['oracle']=[measured('oracle_action',{**base},oracle_action)]
    return {'old_h':old_h,'old_y':old_y,'current_h':current_h,'current_y':current_y,'current_gated':current_gated,'outputs':results,'masks':masks,'mask_guard_host_seconds':guard_wall}
