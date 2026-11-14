"""Causal whole-M proxy primitives, independent of orchestration and GPU setup."""
import torch
import torch.nn.functional as F

def fixed_prefix_permutation(m,seed):
    if m<64:raise ValueError('64 distinct real rows required')
    g=torch.Generator(device='cpu').manual_seed(seed)
    return torch.randperm(m,generator=g)[:64]

def rms_score(current_sample_preact,previous_sample_h):
    if current_sample_preact.ndim!=2 or current_sample_preact.shape!=previous_sample_h.shape or current_sample_preact.shape[0] not in (4,64):raise ValueError('expected 4 or 64 matched real rows')
    if current_sample_preact.dtype!=torch.bfloat16 or previous_sample_h.dtype!=torch.bfloat16:raise ValueError('BF16 operands')
    d=F.gelu(current_sample_preact,approximate='tanh').float()-previous_sample_h.float()
    return d.square().mean(dim=0).sqrt()

def mean_score(current_pre,previous_mean_pre):
    if current_pre.dtype!=torch.bfloat16 or previous_mean_pre.dtype!=torch.bfloat16 or current_pre.shape!=previous_mean_pre.shape:raise ValueError('BF16 operands')
    return (current_pre-previous_mean_pre).abs().float()

def selection(scores,k):
    if scores.ndim!=1 or not torch.isfinite(scores).all() or not 0<=k<=scores.numel():raise ValueError('invalid scores')
    return torch.sort(torch.argsort(scores,descending=True,stable=True)[:k]).values

def online_mask(mode,x,w1,b1,old_h,old_mean,permutation,k,random_indices,return_scores=False):
    """One causal selector; caller measures this together with sparse action."""
    if mode=='mean':
        scores=mean_score(F.linear(x.mean(dim=0),w1,b1),old_mean)
    elif mode in ('p4','p64'):
        rows=permutation[:4 if mode=='p4' else 64].to(device=x.device)
        scores=rms_score(F.linear(x.index_select(0,rows),w1,b1),old_h.index_select(0,rows))
    elif mode=='random':
        mask=random_indices.to(device=x.device,dtype=torch.long);return (mask,None) if return_scores else mask
    else:raise ValueError('unknown online selector')
    mask=selection(scores,k);return (mask,scores) if return_scores else mask

def oracle_mask(current_h,old_h,k,return_scores=False):
    """Post-reference diagnostic only, never callable from online_mask."""
    if current_h.shape!=old_h.shape or current_h.ndim!=2:raise ValueError('shape mismatch')
    d=current_h.float()-old_h.float()
    score=d.square().mean(dim=0).sqrt();mask=selection(score,k);return (mask,score) if return_scores else mask

def execute_pair(x0,x1,w1,b1,w2,b2,gate,permutation,k,random_indices,*,measured,action):
    """Order is part of the contract: all online actions precede full current h.

    measured(name, ledger_debits, thunk) must persist all debit counts before
    timing, wrap only thunk with synchronized CUDA events and host wall, and
    persist results afterwards. action is the real paid selected-W1/W2 path.
    """
    def dense(x):
        pre=F.linear(x,w1,b1);h=F.gelu(pre,approximate='tanh');return pre,h,F.linear(h,w2,b2)
    old_pre,old_h,old_y=measured('old_cache',{'dense_w1':1,'gelu':1,'dense_w2':1,'old_mean':1},lambda:(lambda p,h,y:(p.mean(dim=0),h,y))(*dense(x0)))
    measured('cold_kernel_setup',{'selected_w1':1,'selected_w2':1,'cache_clone':1,'index_cast':1,'delta_allocate':1},lambda:action(random_indices.to(x1.device),old_h,old_y))
    online={};masks={};scores={}
    for mode in ('mean','p4','p64','random'):
        debits={'selected_w1':1,'selected_w2':1,'cache_clone':1,'index_cast':1,'delta_allocate':1,'selector':1,'gate_product':1}
        if mode!='random':debits['proxy_w1']=1
        if mode in ('p4','p64'):debits['proxy_gelu']=1
        def segment(mode=mode):
            mask,score=online_mask(mode,x1,w1,b1,old_h,old_pre,permutation,k,random_indices,return_scores=True)
            return mask,(action(mask,old_h,old_y)*gate).to(torch.bfloat16),score
        masks[mode],online[mode],scores[mode]=measured('online_'+mode,debits,segment)
    def dense_current():
        pre,h,y=dense(x1);return pre,h,y,(y*gate).to(torch.bfloat16)
    _,current_h,current_y,current_gated=measured('dense_reference',{'dense_w1':1,'gelu':1,'dense_w2':1,'gate_product':1},dense_current)
    oracle,oracle_score=measured('oracle_selection',{'oracle_selection':1},lambda:oracle_mask(current_h,old_h,k,return_scores=True))
    masks['oracle']=oracle;scores['oracle']=oracle_score
    online['oracle']=measured('oracle_action',{'selected_w1':1,'selected_w2':1,'cache_clone':1,'index_cast':1,'delta_allocate':1,'gate_product':1},lambda:(action(oracle,old_h,old_y)*gate).to(torch.bfloat16))
    return {'old_h':old_h,'old_y':old_y,'current_h':current_h,'current_y':current_y,'current_gated':current_gated,'scores':scores,'masks':masks,'outputs':online,'gate':gate}
