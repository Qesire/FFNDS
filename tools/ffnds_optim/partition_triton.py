"""Optional single-scan partition and route-grouped selected Triton kernels.

No full score matrix is materialized. Each partition key tile performs one
QK and one PV product, then merges into exactly one of the two states.
These kernels change the frozen backend's finite-precision reduction order.
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _partition_kernel(Q, K, V, ROUTE, MEAN, INVERSE, FULL, SELECTED, RHO,
                      FULL_LSE, SELECTED_LSE, CENTERED,
                      N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                      QSN: tl.constexpr, QSH: tl.constexpr,
                      KSN: tl.constexpr, KSH: tl.constexpr,
                      VSN: tl.constexpr, VSH: tl.constexpr,
                      RSH: tl.constexpr, RSQ: tl.constexpr,
                      TOP: tl.constexpr, ROUTE_PAD: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_K: tl.constexpr, SCALE: tl.constexpr,
                      WRITE_CENTERED: tl.constexpr):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    channel = tl.arange(0, D)
    q = tl.load(Q + rows[:, None]*QSN + head*QSH + channel[None, :],
                mask=rows[:, None] < N, other=0)
    rr = tl.arange(0, ROUTE_PAD)
    route = tl.load(ROUTE + head*RSH + ((tile*BLOCK_M)//BLOCK_SIZE)*RSQ + rr,
                    mask=rr < TOP, other=-1)
    ms = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    mc = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    ls = tl.full((BLOCK_M,), 0., tl.float32)
    lc = tl.full((BLOCK_M,), 0., tl.float32)
    zs = tl.full((BLOCK_M, D), 0., tl.float32)
    zc = tl.full((BLOCK_M, D), 0., tl.float32)
    for start in range(0, tl.cdiv(N, BLOCK_K)):
        key_rows = start*BLOCK_K + tl.arange(0, BLOCK_K)
        kk = tl.load(K + key_rows[None, :]*KSN + head*KSH + channel[:, None],
                     mask=key_rows[None, :] < N, other=0)
        vv = tl.load(V + key_rows[:, None]*VSN + head*VSH + channel[None, :],
                     mask=key_rows[:, None] < N, other=0)
        scores = tl.dot(q, kk, input_precision='ieee')*SCALE
        valid = (rows[:, None] < N) & (key_rows[None, :] < N)
        scores = tl.where(valid, scores, -float('inf'))
        mb = tl.where(rows < N, tl.max(scores, 1), 0.)
        probabilities = tl.where(valid, tl.exp(scores-mb[:, None]), 0.)
        lb = tl.sum(probabilities, 1)
        zb = tl.dot(probabilities.to(vv.dtype), vv, input_precision='ieee')
        chosen = tl.sum((route == ((start*BLOCK_K)//BLOCK_SIZE)).to(tl.int32), 0) > 0
        if chosen:
            mn = tl.maximum(ms, mb)
            old = tl.exp(ms-mn)
            new = tl.exp(mb-mn)
            zs = zs*old[:, None]+zb*new[:, None]
            ls = ls*old+lb*new
            ms = mn
        else:
            mn = tl.maximum(mc, mb)
            old = tl.exp(mc-mn)
            new = tl.exp(mb-mn)
            zc = zc*old[:, None]+zb*new[:, None]
            lc = lc*old+lb*new
            mc = mn
    maximum = tl.maximum(ms, mc)
    ws = tl.where(ls > 0., tl.exp(ms-maximum), 0.)
    wc = tl.where(lc > 0., tl.exp(mc-maximum), 0.)
    denominator = ws*ls+wc*lc
    rrho = wc*lc/tl.maximum(denominator, 1.e-30)
    native = (zs*ws[:, None]+zc*wc[:, None])/tl.maximum(denominator[:, None], 1.e-30)
    selected = zs/tl.maximum(ls[:, None], 1.e-30)
    output_offsets = (head*N+rows[:, None])*D+channel[None, :]
    tl.store(FULL+output_offsets, native, mask=rows[:, None] < N)
    tl.store(SELECTED+output_offsets, selected, mask=rows[:, None] < N)
    tl.store(RHO+head*N+rows, rrho, mask=rows < N)
    tl.store(FULL_LSE+head*N+rows, maximum+tl.log(tl.maximum(denominator, 1.e-30)), mask=rows < N)
    tl.store(SELECTED_LSE+head*N+rows, ms+tl.log(tl.maximum(ls, 1.e-30)), mask=rows < N)
    if WRITE_CENTERED:
        mean = tl.load(MEAN+head*D+channel)
        inverse = tl.load(INVERSE+head*D+channel)
        omitted = zc/tl.maximum(lc[:, None], 1.e-30)
        centered = rrho[:, None]*(omitted-mean[None, :])*inverse[None, :]
        tl.store(CENTERED+output_offsets, centered, mask=rows[:, None] < N)


@triton.jit
def _selected_kernel(Q, K, V, ROUTE, ROW_MAP, ACTIVE_TILES, OUT, LSE,
                     N: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                     D: tl.constexpr, QSN: tl.constexpr, QSH: tl.constexpr,
                     KSN: tl.constexpr, KSH: tl.constexpr,
                     VSN: tl.constexpr, VSH: tl.constexpr,
                     RSH: tl.constexpr, RSQ: tl.constexpr,
                     TOP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                     SCALE: tl.constexpr):
    tile = tl.load(ACTIVE_TILES+tl.program_id(0))
    head = tl.program_id(1)
    global_rows = tile*BLOCK_M+tl.arange(0, BLOCK_M)
    rows = tl.load(ROW_MAP+global_rows)
    valid_q = (rows >= 0) & (global_rows < N)
    channel = tl.arange(0, D)
    q = tl.load(Q+rows[:, None]*QSN+head*QSH+channel[None, :], mask=valid_q[:, None], other=0)
    maximum = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    den = tl.full((BLOCK_M,), 0., tl.float32)
    acc = tl.full((BLOCK_M, D), 0., tl.float32)
    for route_index in range(0, TOP):
        key_block = tl.load(ROUTE+head*RSH+((tile*BLOCK_M)//BLOCK_SIZE)*RSQ+route_index)
        for chunk in range(0, BLOCK_SIZE//BLOCK_K):
            key_rows = key_block*BLOCK_SIZE+chunk*BLOCK_K+tl.arange(0, BLOCK_K)
            kk = tl.load(K+key_rows[None, :]*KSN+head*KSH+channel[:, None], mask=key_rows[None, :] < N, other=0)
            vv = tl.load(V+key_rows[:, None]*VSN+head*VSH+channel[None, :], mask=key_rows[:, None] < N, other=0)
            scores = tl.dot(q, kk, input_precision='ieee')*SCALE
            valid = valid_q[:, None] & (key_rows[None, :] < N)
            scores = tl.where(valid, scores, -float('inf'))
            block_max = tl.where(valid_q, tl.max(scores, 1), 0.)
            merged = tl.maximum(maximum, block_max)
            alpha = tl.exp(maximum-merged)
            p = tl.where(valid, tl.exp(scores-merged[:, None]), 0.)
            acc = acc*alpha[:, None]+tl.dot(p.to(vv.dtype), vv, input_precision='ieee')
            den = den*alpha+tl.sum(p, 1)
            maximum = merged
    result = acc/tl.maximum(den[:, None], 1.e-30)
    tl.store(OUT+(head*NQ+rows[:, None])*D+channel[None, :], result, mask=valid_q[:, None])
    tl.store(LSE+head*NQ+rows, maximum+tl.log(tl.maximum(den, 1.e-30)), mask=valid_q)


def _check(q, k, v, block_size, query_tile, key_tile):
    interpreted = os.environ.get('TRITON_INTERPRET') == '1'
    if not q.is_cuda and not interpreted:
        raise ValueError('Triton execution needs CUDA; use backend="torch" for CPU reference')
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Triton kernels support FP16/BF16/FP32 inputs')
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError('Supported head dimensions: 16,32,64,128')
    if block_size % query_tile or block_size % key_tile:
        raise ValueError('Triton tiles must divide the original route block')
    if query_tile < 16 or key_tile < 16:
        raise ValueError('Dot-product tiles must be at least 16')
    if any(t.stride(-1) != 1 for t in (q,k,v)):
        raise ValueError('Last Q/K/V channel axis must be contiguous')


def partition_triton(q,k,v,route_indices, *, block_size=128,query_tile=32,
                     key_tile=64,mean=None,inverse=None):
    from .attention import PartitionOutput
    _check(q,k,v,block_size,query_tile,key_tile)
    n,h,d=q.shape
    shape=(h,n,d)
    full=torch.empty(shape,device=q.device,dtype=q.dtype)
    selected=torch.empty_like(full)
    rho=torch.empty((h,n,1),device=q.device,dtype=torch.float32)
    full_lse=torch.empty((h,n),device=q.device,dtype=torch.float32)
    selected_lse=torch.empty_like(full_lse)
    centered=torch.empty_like(full) if mean is not None else None
    mean_arg=mean.contiguous() if mean is not None else q
    inv_arg=inverse.contiguous() if inverse is not None else q
    _partition_kernel[(triton.cdiv(n,query_tile),h)](
        q,k,v,route_indices,mean_arg,inv_arg,full,selected,rho,full_lse,selected_lse,
        centered if centered is not None else full,
        n,h,d,*q.stride()[:2],*k.stride()[:2],*v.stride()[:2],
        *route_indices.stride()[:2],route_indices.shape[-1],triton.next_power_of_2(route_indices.shape[-1]),
        block_size,query_tile,key_tile,d**-.5,centered is not None,
        num_warps=4,num_stages=2,enable_fp_fusion=True)
    return PartitionOutput(full,selected,rho,full_lse,selected_lse,centered)


def selected_triton(q,k,v,route_indices,layout, *,key_tile=64):
    _check(q,k,v,layout.block_size,layout.query_tile,key_tile)
    nq,h,d=q.shape
    out=torch.empty((h,nq,d),device=q.device,dtype=q.dtype)
    lse=torch.empty((h,nq),device=q.device,dtype=torch.float32)
    if nq:
        _selected_kernel[(len(layout.active_tiles),h)](
            q,k,v,route_indices,layout.row_map,layout.active_tiles,out,lse,
            len(k),nq,h,d,*q.stride()[:2],*k.stride()[:2],*v.stride()[:2],
            *route_indices.stride()[:2],route_indices.shape[-1],layout.block_size,
            layout.query_tile,key_tile,d**-.5,num_warps=4,num_stages=2,
            enable_fp_fusion=True)
    return out,lse
