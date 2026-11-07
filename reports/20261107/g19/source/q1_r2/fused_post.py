"""Post-residual fusion preserving the native BF16 rounding boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x, y, q, cache, gate, n:tl.constexpr, d:tl.constexpr, ACTION:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    m = i < n
    xv = tl.load(x+i,m,other=0).to(tl.float32)
    g = tl.load(gate+i%d,m,other=0).to(tl.float32)
    if ACTION == 0:
        # anchor: retain native y; separately encode BF16(y-q) into cache
        v = tl.load(y+i,m,other=0).to(tl.float32)
        pred = tl.load(q+i,m,other=0).to(tl.float32)
        tl.store(cache+i,v-pred,m)
    elif ACTION == 1:
        pred = tl.load(q+i,m,other=0).to(tl.float32)
        c = tl.load(cache+i,m,other=0).to(tl.float32)
        # Round replacement output to BF16 before multiplying by gate.
        v = (pred+c).to(tl.bfloat16).to(tl.float32)
    else:
        # parallel anchor post: y is native, cache is written on auxiliary stream.
        v = tl.load(y+i,m,other=0).to(tl.float32)
    gated = (v*g).to(tl.bfloat16).to(tl.float32)
    tl.store(x+i,xv+gated,m)


def run(x, y, q, cache, gate, *, action):
    if action not in ('anchor','hit','post'): raise ValueError('action')
    inputs = [x,gate] + ([y] if action in ('anchor','post') else []) + ([q,cache] if action != 'post' else [])
    if any(t is None or t.dtype != torch.bfloat16 or t.device.type != 'cuda' or not t.is_contiguous() or t.device != x.device for t in inputs):
        raise RuntimeError('requires contiguous same-device BF16 tensors')
    if x.numel() % 5120 or gate.numel() != 5120 or (y is not None and y.numel()!=x.numel()) or (q is not None and q.numel()!=x.numel()) or (cache is not None and cache.numel()!=x.numel()):
        raise RuntimeError('invalid FFN/gate geometry')
    # Unused pointers in the selected static branch may safely alias x.
    n=x.numel(); block=1024
    _kernel[(triton.cdiv(n,block),)](x, y if y is not None else x, q if q is not None else x,
                                    cache if cache is not None else x, gate, n,5120,
                                    ACTION={'anchor':0,'hit':1,'post':2}[action], BLOCK=block,
                                    enable_fp_fusion=False)


def validate_synthetic():
    """Exact bit comparison on CUDA after compute-node/scope guard."""
    gen = torch.Generator(device='cpu').manual_seed(113)
    def value(n): return torch.randn(n,generator=gen).to(torch.bfloat16).cuda()
    x,y,q,c,g = value(3*5120),value(3*5120),value(3*5120),value(3*5120),value(5120)
    # Counterexample to merging the gate multiply with the residual addition.
    x[0]=-1.015625; y[0]=1.0078125; g[0]=1.0078125
    # Counterexample to skipping the BF16 hit-output rounding boundary.
    q[1]=1.0; c[1]=0.00390625
    x0=x.clone(); c0=c.clone()
    run(x,y,q,c,g,action='anchor')
    reference_c=(y.float()-q.float()).to(torch.bfloat16)
    reference_x=(x0.float()+(y*g.repeat(3)).float()).to(torch.bfloat16)
    if not torch.equal(c.view(torch.int16),reference_c.view(torch.int16)) or not torch.equal(x.view(torch.int16),reference_x.view(torch.int16)):
        raise RuntimeError('anchor post fusion bit mismatch')
    x=x0.clone(); run(x,None,q,c0,g,action='hit')
    yh=(q.float()+c0.float()).to(torch.bfloat16)
    reference_x=(x0.float()+(yh*g.repeat(3)).float()).to(torch.bfloat16)
    if not torch.equal(x.view(torch.int16),reference_x.view(torch.int16)):
        raise RuntimeError('hit post fusion bit mismatch')
    x=x0.clone(); run(x,y,None,None,g,action='post')
    reference_x=(x0.float()+(y*g.repeat(3)).float()).to(torch.bfloat16)
    if not torch.equal(x.view(torch.int16),reference_x.view(torch.int16)):
        raise RuntimeError('parallel anchor post fusion bit mismatch')
    torch.cuda.synchronize()
    return {'status':'PASS','elements':x.numel(),'comparison':'BF16 bit patterns; anchor, hit, post'}
