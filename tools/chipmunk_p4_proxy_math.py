"""Causal score primitives for P4. Callers own all paid operators and caches."""
import torch
import torch.nn.functional as F

def select(scores, k):
    if scores.ndim != 1 or not torch.isfinite(scores).all() or not 0 <= k <= scores.numel():
        raise ValueError('invalid finite score vector or K')
    return torch.sort(torch.argsort(scores,descending=True,stable=True)[:k]).values

def sample_rms_score(current_four_preact, previous_four_h):
    """Only four paid current W1 rows and corresponding old cache are visible."""
    if current_four_preact.ndim != 2 or current_four_preact.shape != previous_four_h.shape or current_four_preact.shape[0] != 4:
        raise ValueError('four matched rows required')
    if current_four_preact.dtype != torch.bfloat16 or previous_four_h.dtype != torch.bfloat16:
        raise ValueError('BF16 operands required')
    now=F.gelu(current_four_preact,approximate='tanh')
    diff=now.float()-previous_four_h.float()
    return torch.sqrt(torch.mean(diff*diff,dim=0))

def mean_score(current_mean_preact, previous_mean_preact):
    if current_mean_preact.shape != previous_mean_preact.shape or current_mean_preact.dtype != torch.bfloat16 or previous_mean_preact.dtype != torch.bfloat16:
        raise ValueError('BF16 matching operands required')
    return (current_mean_preact-previous_mean_preact).abs().float()

def pair_core(x_previous, x_current, w1, b1, w2, b2, sample_rows, random_indices, k, *, charged, action):
    """Invoke all numerical operations through charged(name, thunk).

    action(mask, old_h, old_y) is the paid physical sparse action, called
    only after all three selections are finalized. No full-current operator is
    executed before the selections. Caller supplies synchronization/timing,
    finite checks, fail-closed accounting and dtype/shape checks.
    """
    n=x_previous.shape[0];h=w1.shape[0]
    if x_previous.shape!=x_current.shape or x_previous.ndim!=2 or len(sample_rows)!=4 or len(set(sample_rows))!=4 or any(not 0<=i<n for i in sample_rows):
        raise ValueError('invalid paired coordinates')
    if len(random_indices)!=k or len(set(random_indices))!=k or any(not 0<=i<h for i in random_indices):raise ValueError('invalid fixed random control')
    old_pre=charged('previous_w1',lambda:F.linear(x_previous,w1,b1))
    old_h=charged('previous_gelu',lambda:F.gelu(old_pre,approximate='tanh'))
    old_y=charged('previous_w2',lambda:F.linear(old_h,w2,b2))
    old_mean=charged('old_preact_mean',lambda:old_pre.mean(dim=0))
    current_mean=charged('current_input_mean',lambda:x_current.mean(dim=0))
    proxy_pre=charged('mean_w1',lambda:F.linear(current_mean,w1,b1))
    mean_s=charged('mean_score',lambda:mean_score(proxy_pre,old_mean))
    four_x=charged('sample_gather',lambda:x_current[sample_rows])
    four_old=charged('old_cache_gather',lambda:old_h[sample_rows])
    sample_pre=charged('sample_w1',lambda:F.linear(four_x,w1,b1))
    sample_s=charged('sample_score_gelu_rms',lambda:sample_rms_score(sample_pre,four_old))
    masks=charged('selection',lambda:{'mean':select(mean_s,k),'sample_rms':select(sample_s,k),'random':torch.tensor(sorted(random_indices),device=x_current.device,dtype=torch.long)})
    results={name:charged('sparse_action_'+name,lambda mask=mask:action(mask,old_h,old_y)) for name,mask in masks.items()}
    current_pre=charged('reference_w1_after_selections',lambda:F.linear(x_current,w1,b1))
    current_h=charged('reference_gelu',lambda:F.gelu(current_pre,approximate='tanh'))
    current_y=charged('reference_w2',lambda:F.linear(current_h,w2,b2))
    return {'masks':masks,'mean_scores':mean_s,'sample_scores':sample_s,'previous_h':old_h,'previous_y':old_y,'current_h_reference':current_h,'current_y_reference':current_y,'actions':results}
