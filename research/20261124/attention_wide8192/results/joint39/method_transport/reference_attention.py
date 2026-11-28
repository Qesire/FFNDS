"""CPU mathematical reference for grouped-current attention and defect recovery.

Axes are H (heads), I (queries), J (keys), D (key/value channels). Inputs are
effective Q/K after norm/RoPE, and current V. Arithmetic/state use float64 for
mathematical verification; this is not a BF16 implementation or a GPU kernel.

Only selected exact rows and small coarse chunks are materialized. The hit
teacher receives exactly the requested P rows and all current K/V. An optional
transported_anchor_defect lets a separately specified transport act on the
existing field without changing the current feature/teacher definitions.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Config:
    key_block: int = 128
    query_block: int = 128
    query_chunk: int = 64
    coarse_chunk: int = 32
    feature_channels: int = 15
    ridge_per_probe: float = 0.1
    scale_floor: float = 1e-12


@dataclass
class GroupSummary:
    counts: np.ndarray       # [H, key_blocks, 2], actual valid keys
    key_mean: np.ndarray     # [H, key_blocks, 2, Dk]
    value_mean: np.ndarray   # [H, key_blocks, 2, Dv]


@dataclass
class BaseResult:
    output: np.ndarray      # G [H,I,Dv]
    selected: np.ndarray    # B [H,I,Dv]
    lse: np.ndarray          # log Z0 [H,I]
    selected_lse: np.ndarray # log Zs [H,I]
    groups: GroupSummary


@dataclass
class AnchorState:
    defect: np.ndarray      # Raw E_a or normalized C_a=E_a/s_a, full [H,I,Dv]
    labels: np.ndarray      # [H,J], uint8 membership, frozen until next anchor
    selected_blocks: np.ndarray # [H,query_blocks,key_blocks], frozen support
    config: Config
    cache_mode: str = 'channel_normalized' # raw, scalar_normalized, channel_normalized


@dataclass
class HitResult:
    output: np.ndarray
    base: BaseResult
    features: np.ndarray
    coefficients: np.ndarray
    teacher: np.ndarray | None
    probe_residual: np.ndarray
    anchor_term: np.ndarray
    value_scale: np.ndarray
    constant_value_heads: np.ndarray
    constant_value_channels: np.ndarray


def operands(q, k, v):
    q, k, v = [np.asarray(x, dtype=np.float64) for x in (q, k, v)]
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError('Expected H,I,D / H,J,D arrays')
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Head/key/channel axes disagree')
    if min(q.shape+k.shape+v.shape) < 1 or not all(np.isfinite(x).all() for x in (q,k,v)):
        raise ValueError('Operands must be nonempty and finite')
    return q,k,v


def full_attention(q, k, v, query_chunk=64):
    """Stable exact reference over supplied queries; chunks avoid I x J storage."""
    q,k,v=operands(q,k,v)
    out=np.empty((q.shape[0],q.shape[1],v.shape[-1]),dtype=np.float64)
    lse=np.empty(q.shape[:2],dtype=np.float64)
    for h in range(q.shape[0]):
        for i in range(0,q.shape[1],query_chunk):
            scores=q[h,i:i+query_chunk]@k[h].T/np.sqrt(k.shape[-1])
            m=scores.max(1)
            w=np.exp(scores-m[:,None])
            z=w.sum(1)
            out[h,i:i+query_chunk]=(w@v[h])/z[:,None]
            lse[h,i:i+query_chunk]=m+np.log(z)
    return out,lse


def group_labels(k, direction, key_block=128):
    """One centered sign split in each original key block, independently per head."""
    k=np.asarray(k,dtype=np.float64)
    r=np.asarray(direction,dtype=np.float64)
    if r.shape!=(k.shape[-1],) or np.linalg.norm(r)==0:
        raise ValueError('A nonzero key-channel direction is required')
    r=r/np.linalg.norm(r)
    labels=np.empty(k.shape[:2],dtype=np.uint8)
    for start in range(0,k.shape[1],key_block):
        block=k[:,start:start+key_block]
        labels[:,start:start+key_block]=((block-block.mean(1,keepdims=True))@r>=0)
    return labels


def summarize(k,v,labels,key_block=128):
    k,v=np.asarray(k,dtype=np.float64),np.asarray(v,dtype=np.float64)
    labels=np.asarray(labels)
    if labels.shape!=k.shape[:2] or not np.isin(labels,[0,1]).all():
        raise ValueError('Frozen labels must assign each valid key to one of two groups')
    H,J,D=k.shape
    blocks=(J+key_block-1)//key_block
    n=np.zeros((H,blocks,2),dtype=np.int64)
    km=np.zeros((H,blocks,2,D),dtype=np.float64)
    vm=np.zeros((H,blocks,2,v.shape[-1]),dtype=np.float64)
    for h in range(H):
        for b,start in enumerate(range(0,J,key_block)):
            ids=np.arange(start,min(start+key_block,J))
            for side in (0,1):
                keep=ids[labels[h,ids]==side]
                n[h,b,side]=len(keep)
                if len(keep):
                    km[h,b,side]=k[h,keep].mean(0)
                    vm[h,b,side]=v[h,keep].mean(0)
    return GroupSummary(n,km,vm)


def grouped_attention(q,k,v,selected_blocks,labels,config=Config()):
    """Exact selected blocks + two positive summaries for each omitted block.

    A selected block contributes its keys exactly once. Its summaries are
    excluded. Empty groups are skipped and the final partial block uses only
    valid keys. Coarse log-mass chunks are merged by online log-sum-exp.
    """
    q,k,v=operands(q,k,v)
    H,I,D=q.shape; J=k.shape[1]
    qb=(I+config.query_block-1)//config.query_block
    kb=(J+config.key_block-1)//config.key_block
    support=np.asarray(selected_blocks,dtype=bool)
    if support.shape!=(H,qb,kb) or not support.any(2).all():
        raise ValueError('Every query block needs a nonempty selected block set')
    sums=summarize(k,v,labels,config.key_block)
    G=np.empty((H,I,v.shape[-1]),dtype=np.float64)
    B=np.empty_like(G)
    lse0=np.empty((H,I)); lses=np.empty_like(lse0)
    parents=np.arange(J)//config.key_block
    for h in range(H):
        for b in range(qb):
            ids=np.flatnonzero(support[h,b,parents])
            coarse=np.argwhere((~support[h,b,:,None]) & (sums.counts[h]>0))
            cm=sums.key_mean[h,coarse[:,0],coarse[:,1]]
            vm=sums.value_mean[h,coarse[:,0],coarse[:,1]]
            logn=np.log(sums.counts[h,coarse[:,0],coarse[:,1]])
            for start in range(b*config.query_block,min((b+1)*config.query_block,I),config.query_chunk):
                stop=min(start+config.query_chunk,(b+1)*config.query_block,I)
                qs=q[h,start:stop]
                selected_scores=qs@k[h,ids].T/np.sqrt(D)
                shift=selected_scores.max(1)
                sw=np.exp(selected_scores-shift[:,None]); sz=sw.sum(1)
                B[h,start:stop]=sw@v[h,ids]/sz[:,None]
                lses[h,start:stop]=shift+np.log(sz)
                m=lses[h,start:stop].copy()
                den=np.ones(stop-start)
                num=B[h,start:stop].copy()
                for c in range(0,len(coarse),config.coarse_chunk):
                    logits=qs@cm[c:c+config.coarse_chunk].T/np.sqrt(D)+logn[c:c+config.coarse_chunk]
                    m1=np.maximum(m,logits.max(1))
                    old=np.exp(m-m1); w=np.exp(logits-m1[:,None])
                    num=num*old[:,None]+w@vm[c:c+config.coarse_chunk]
                    den=den*old+w.sum(1); m=m1
                G[h,start:stop]=num/den[:,None]
                lse0[h,start:stop]=m+np.log(den)
    return BaseResult(G,B,lse0,lses,sums)


def value_scale(v,config=Config(),mode='channel_normalized'):
    """Return positive solve scales [H,D] and exact constant-channel flags.

    Channel mode uses std over keys. Scalar/raw mode broadcasts one centered
    head RMS. A finite floor defines numerical behavior; affine-equivariance
    claims require that the floor is inactive on nonconstant channels.
    """
    constant=(v==v[:,:1,:]).all(axis=1)
    centered=v-v.mean(1,keepdims=True)
    if mode=='channel_normalized':
        scale=np.sqrt(np.mean(centered**2,axis=1))
        scale[constant]=0.
    else:
        rms=np.sqrt(np.mean(centered**2,axis=(1,2)))
        rms[constant.all(1)]=0.
        scale=np.broadcast_to(rms[:,None],constant.shape).copy()
    return np.maximum(scale,config.scale_floor),constant


def current_features(q,v,base,pk,pv,config=Config(),mode='channel_normalized'):
    """32 current columns at r=15; no encoded anchor field enters features."""
    r=config.feature_channels
    pk,pv=np.asarray(pk,dtype=np.float64),np.asarray(pv,dtype=np.float64)
    if pk.shape[0]!=q.shape[-1] or pv.shape[0]!=v.shape[-1] or min(pk.shape[1],pv.shape[1])<r:
        raise ValueError('Projection assets do not cover requested channel features')
    scale,constant=value_scale(v,config,mode)
    response=(base.output-base.selected)/scale[:,None,:]
    response=np.where(constant[:,None,:],0.,response)
    psi=np.concatenate([
        np.ones(q.shape[:2]+(1,)),
        q@pk[:,:r]/np.sqrt(q.shape[-1]),
        response@pv[:,:r],
        np.exp(base.selected_lse-base.lse)[...,None]
    ],axis=-1)
    return psi,scale,constant


def build_anchor(q,k,v,selected_blocks,pk,pv,config=Config(),native_output=None,cache_mode='channel_normalized'):
    """Uses the full native anchor already available in production when supplied."""
    q,k,v=operands(q,k,v)
    labels=group_labels(k,np.asarray(pk)[:,0],config.key_block)
    base=grouped_attention(q,k,v,selected_blocks,labels,config)
    if native_output is None:
        native_output,_=full_attention(q,k,v,config.query_chunk)
    native_output=np.asarray(native_output,dtype=np.float64)
    if native_output.shape!=base.output.shape:
        raise ValueError('Native anchor output has incorrect axes')
    if cache_mode not in ('raw','scalar_normalized','channel_normalized'):
        raise ValueError('Unsupported cache coordinate mode')
    cache=native_output-base.output
    if cache_mode!='raw':
        scale,constant=value_scale(v,config,cache_mode)
        cache=cache/scale[:,None,:]
        cache=np.where(constant[:,None,:],0.,cache)
    state=AnchorState(cache,labels.copy(),np.array(selected_blocks,dtype=bool,copy=True),config,cache_mode)
    # base.groups exposes anchor value means; it is not retained in minimal state.
    return state,base,native_output


def recover_hit(q,k,v,pk,pv,probes,state=None,selected_blocks=None,config=Config(),
                teacher:Callable|None=None,transported_anchor_defect=None,cache_mode='channel_normalized'):
    """Current base, pre-overwrite labels, ridge recovery, final exact P overwrite.

    If state is absent, selected_blocks is required and current labels are made;
    the target is E_s. With state, labels/support are frozen and the target is
    E_s-E_a. Normalized cache uses C_a D_s and targets (A_s-G_s)D_s^+-C_a.
    For constant-current-V columns, G is already exact and history is bypassed.
    Cache quantization and BF16 solve/cast order are not modeled here.
    """
    q,k,v=operands(q,k,v)
    P=np.asarray(probes,dtype=np.int64)
    if P.ndim!=1 or len(P)==0 or len(np.unique(P))!=len(P) or P.min()<0 or P.max()>=q.shape[1]:
        raise ValueError('P must contain distinct valid query row indices')
    if state is None:
        if selected_blocks is None:
            raise ValueError('Unanchored call requires its selected block support')
        labels=group_labels(k,np.asarray(pk)[:,0],config.key_block)
        anchor=np.zeros((q.shape[0],q.shape[1],v.shape[-1]))
    else:
        config=state.config
        cache_mode=state.cache_mode
        labels=state.labels
        selected_blocks=state.selected_blocks
        anchor=state.defect
        if anchor.shape!=(q.shape[0],q.shape[1],v.shape[-1]):
            raise ValueError('Anchor logical query/value axes changed')
    if transported_anchor_defect is not None:
        if state is None:
            raise ValueError('A transported defect requires anchor state')
        anchor=np.asarray(transported_anchor_defect,dtype=np.float64)
        if anchor.shape!=state.defect.shape:
            raise ValueError('Transported defect axes disagree')
    base=grouped_attention(q,k,v,selected_blocks,labels,config)
    psi,scale,constant=current_features(q,v,base,pk,pv,config,cache_mode)
    anchor=np.array(anchor,copy=True)
    if state is not None and state.cache_mode!='raw' and transported_anchor_defect is None:
        anchor=anchor*scale[:,None,:]
    anchor=np.where(constant[:,None,:],0.,anchor)
    before=base.output+anchor
    W=np.zeros((q.shape[0],psi.shape[-1],v.shape[-1]))
    residual=np.zeros((q.shape[0],len(P),v.shape[-1]))
    exact=None
    if not constant.all():
        if teacher is None:
            exact,_=full_attention(q[:,P],k,v,config.query_chunk)
        else:
            exact=np.asarray(teacher(q[:,P],k,v),dtype=np.float64)
        if exact.shape!=residual.shape:
            raise ValueError('P teacher output axes disagree')
        residual=exact-before[:,P]  # Form observations before any native overwrite.
        for h in np.flatnonzero(~constant.all(1)):
            xp=psi[h,P]
            gram=xp.T@xp+len(P)*config.ridge_per_probe*np.eye(psi.shape[-1])
            normalized=residual[h]/scale[h]
            normalized[:,constant[h]]=0.
            W[h]=np.linalg.solve(gram,xp.T@normalized)
    output=before+scale[:,None,:]*(psi@W)
    output=np.where(constant[:,None,:],base.output,output)
    if exact is not None:
        output[:,P]=exact
    return HitResult(output,base,psi,W,exact,residual,anchor,scale,constant.all(1),constant)


def defect_projection(state,projection):
    """Optional anchor dictionary; computing it does not alter minimal state."""
    return state.defect@np.asarray(projection,dtype=np.float64)
