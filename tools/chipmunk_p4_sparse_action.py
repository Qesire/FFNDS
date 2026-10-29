"""P4's paid A100 selected-delta action; uses existing reviewed kernels."""
import torch

def selected_action(x,w1,b1,w2,old_h,old_y,indices,*,charged,kernels):
    """Indices chosen earlier; clones, gather payload and both GPU calls are paid."""
    m,d=x.shape; h=w1.shape[0]; k=indices.numel()
    if x.dtype!=torch.bfloat16 or w1.shape!=(h,d) or w2.shape!=(d,h) or old_h.shape!=(m,h) or old_y.shape!=(m,d) or b1.shape!=(h,) or indices.ndim!=1 or k==0:
        raise ValueError('invalid sparse action shape')
    def copy():return old_h.clone(),old_y.clone()
    hc,yc=charged('action_cache_clone',copy)
    ids=charged('action_index_cast',lambda:indices.to(dtype=torch.int32).contiguous())
    delta=charged('action_delta_allocate',lambda:torch.empty((m,k),dtype=torch.bfloat16,device=x.device))
    ga=((m+15)//16,(k+31)//32); gb=((m+15)//16,(d+31)//32)
    charged('action_W1_GELU',lambda:kernels.selected_w1_gelu_delta[ga](x,w1,b1,hc,ids,delta,m,d,h,k,16,32,32))
    charged('action_W2_cache_add',lambda:kernels.selected_w2_add_cache[gb](delta,w2,ids,yc,m,d,h,k,16,32,32))
    return yc
