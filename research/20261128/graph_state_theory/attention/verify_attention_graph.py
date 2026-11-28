"""CPU algebra witnesses: actual frozen CAL bases, synthetic operands, zero GPU/model runs."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np

def bf16(x):
    a=np.asarray(x,dtype=np.float32)
    bits=a.view(np.uint32).copy()
    bits += np.uint32(0x7FFF)+(bits>>16 & 1)
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32).astype(np.float64)

def softmax(x):
    e=np.exp(x-np.max(x,axis=-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def lse(x):
    m=np.max(x,axis=-1);return m+np.log(np.exp(x-m[...,None]).sum(-1))

def err(x):return float(np.max(np.abs(x)))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cal-root',type=Path,default=Path('current_review_math/replay'))
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('ATTENTION_GRAPH_VERIFIED.json'))
    args=parser.parse_args();rng=np.random.default_rng(20261128)
    H,n,d,r,p=2,257,128,31,64
    pv=np.load(args.cal_root/'plan_assets/bases/P_v.npy')[:,:64].astype(float)
    reports={}
    for expert,key in [('high','out_current_basis'),('low','out_delta_basis')]:
        path=args.cal_root/'calibration'/f'self_{expert}_transform.npz'
        with np.load(path) as data:basis=data[key].astype(float)
        B=pv@basis;sv=np.linalg.svd(B,compute_uv=False)
        reports[expert]=dict(shape=list(B.shape),rank=int(np.linalg.matrix_rank(B)),
              sigma_max=float(sv[0]),sigma_min=float(sv[-1]),condition=float(sv[0]/sv[-1]),
              transform_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with np.load(args.cal_root/'calibration/self_low_transform.npz') as data:basis=data['out_delta_basis'].astype(float)
    B=pv@basis
    q,k,v=[rng.normal(size=(H,n,d)) for _ in range(3)]
    logits=q@k.swapaxes(-1,-2)/d**.5
    idx=np.arange(n);keep=((idx[None,:]//32-idx[:,None]//32)%9)<3
    prob=softmax(logits);full=prob@v
    ls=np.where(keep,logits,-np.inf);lu=np.where(~keep,logits,-np.inf)
    selected=softmax(ls)@v;omitted=softmax(lu)@v
    lses,lseu=lse(ls),lse(lu);lset=np.logaddexp(lses,lseu);rho=np.exp(lseu-lset)
    full_partition=(1-rho)[...,None]*selected+rho[...,None]*omitted
    mu=v.mean(1,keepdims=True);sigma=v.std(1,keepdims=True)
    C=(full-(1-rho)[...,None]*selected-rho[...,None]*mu)/sigma
    C_from_partition=rho[...,None]*(omitted-mu)/sigma
    partition=dict(full_max_abs=err(full_partition-full),rho_max_abs=err(rho-np.where(~keep,prob,0).sum(-1)),centered_max_abs=err(C-C_from_partition))
    m=[np.full((H,n,1),-np.inf) for _ in range(2)]
    den=[np.zeros((H,n,1)) for _ in range(2)];num=[np.zeros((H,n,d)) for _ in range(2)]
    for start in range(0,n,32):
        stop=min(n,start+32);block=logits[...,start:stop];mb=block.max(-1,keepdims=True)
        eb=np.exp(block-mb);lb=eb.sum(-1,keepdims=True);zb=eb@v[:,start:stop]
        flag=keep[:,start][None,:,None]
        for part in range(2):
            active=flag if part==0 else ~flag
            merged=np.maximum(m[part],mb);oldscale=np.exp(m[part]-merged);blockscale=np.exp(mb-merged)
            den[part]=np.where(active,oldscale*den[part]+blockscale*lb,den[part])
            num[part]=np.where(active,oldscale*num[part]+blockscale*zb,num[part])
            m[part]=np.where(active,merged,m[part])
    mfull=np.maximum(m[0],m[1]);w=[np.exp(m[i]-mfull) for i in range(2)]
    ostream=(w[0]*num[0]+w[1]*num[1])/(w[0]*den[0]+w[1]*den[1])
    partition['single_key_scan_max_abs']=err(ostream-full)
    S=bf16(selected+.08*rng.normal(size=selected.shape));history=bf16(C)
    anchor=bf16((selected@pv)@basis);features=bf16((S@pv)@basis)
    x=np.concatenate([np.ones((H,n,1)),features-anchor],-1)
    sigma_t=sigma*(1+.05*rng.normal(size=sigma.shape));mu_t=mu+.03*rng.normal(size=mu.shape)
    base=(1-rho[...,None])*S+rho[...,None]*mu_t+history*sigma_t
    ids=np.arange(p);truth=full[:,:p]+.1*rng.normal(size=(H,p,d));xp=x[:,ids];alpha=6.4
    pinv=np.linalg.solve(xp.swapaxes(-1,-2)@xp+alpha*np.eye(r+1),xp.swapaxes(-1,-2))
    coef=pinv@(truth-base[:,ids]);dense=base+x@coef;dense[:,ids]=truth
    chunks=[]
    for start in range(0,n,48):
        end=min(n,start+48);Ss=S[:,start:end]
        xf=np.concatenate([np.ones((H,end-start,1)),bf16((Ss@pv)@basis)-anchor[:,start:end]],-1)
        bs=(1-rho[:,start:end,None])*Ss+rho[:,start:end,None]*mu_t+history[:,start:end]*sigma_t
        chunks.append(bs+xf@coef)
    tiled=np.concatenate(chunks,1);tiled[:,ids]=truth
    streaming=dict(output_max_abs=err(tiled-dense),output_bf16_equal=bool(np.array_equal(bf16(tiled),bf16(dense))))
    with np.load(args.cal_root/'calibration/self_high_transform.npz') as data: Bhigh=pv@data['out_current_basis'].astype(float)
    W=rng.normal(size=(H,d,256));b=rng.normal(size=(H,1,d))*.01;gamma=rng.normal(size=(H,r,d))*.01
    raw=(S+(S@Bhigh)@gamma+b)@W;effective=W+Bhigh[None]@(gamma@W);folded=S@effective+b@W
    rounded=(S+bf16(S@Bhigh)@gamma+b)@W;qerr=(bf16(S@Bhigh)-S@Bhigh)@gamma@W
    fold=dict(real_max_abs=err(raw-folded),feature_rounding_defect_identity_max_abs=err(rounded-folded-qerr))
    P=np.zeros((p,n));P[np.arange(p),ids]=1;Hlift=x@pinv@P;L=np.eye(n)[None]-Hlift
    eig=np.linalg.eigvals(L[0]);Q=np.eye(n);Q[ids,ids]=0;post=Q@L[0]
    ridge=dict(pre_overwrite_rank=int(np.linalg.matrix_rank(L[0])),n=n,min_real_eigenvalue=float(eig.real.min()),
       post_overwrite_rank=int(np.linalg.matrix_rank(post)),expected_post_rank=n-p,intercept_regularized=True,alpha=alpha)
    rank=64;z=np.concatenate([history,anchor],-1);R,_=np.linalg.qr(rng.normal(size=(d+r,rank)))
    zhat=(z@R)@R.T;zhat[:,ids]=z[:,ids]
    diagonals=np.stack([np.diag(vv) for vv in sigma_t[:,0]])
    G=np.concatenate([diagonals,-coef[:,1:]],1);historical=z@G;direct=(z@R)@(R.T[None]@G);direct[:,ids]=historical[:,ids]
    error=(zhat-z)@G
    compression=dict(readout_identity_max_abs=err(direct-historical-error),probes_exact=bool(np.array_equal(zhat[:,ids],z[:,ids])),
       note='Synthetic operands; error formula only, not a useful rank or Wan quality.')
    # True production shapes: all FLOPs count multiply-add as 2 operations.
    N,HH,D,DD,Llayers,RR,Pn=31668,40,128,5120,40,31,64
    projected_two_stage=2*N*HH*(D*64+64*RR)
    projected_fused=2*N*HH*D*RR
    readout=2*N*HH*(RR+1)*D
    prep=4*HH*D*RR*DD+2*HH*D*DD
    dense_attention=4*N*N*HH*D
    selected_attention=4*N*(31*128)*HH*D
    costs=dict(shapes=dict(N=N,H=HH,d=D,model_width=DD,layers=Llayers,rank=RR,P=Pn),
       high_feature_current_gflop=projected_two_stage/1e9,
       high_feature_composed_basis_gflop=projected_fused/1e9,
       high_readout_current_gflop=readout/1e9,
       high_dynamic_Wo_preparation_gflop=prep/1e9,
       high_Wo_dense_gflop=2*N*DD*DD/1e9,
       high_probe_native_Wo_gflop=2*Pn*DD*DD/1e9,
       high_probe_cost_scope="P64 uses original Wo on teacher rows; if main GEMM skips P, it replaces rather than adds these rows. If full N GEMM is kept, count this as added work.",
       high_Wo_eff_bf16_MiB=2*DD*DD/2**20,
       high_saved_vs_current_recovery_gflop_per_layer=(projected_two_stage+readout-prep)/1e9,
       selected_probe_extra_work_fraction=Pn/N,
       low_anchor_repeated_selected_Tflop_per_layer=selected_attention/1e12,
       low_anchor_repeated_selected_Tflop_40layers=selected_attention*Llayers/1e12,
       note='Selected support cost assumes 3968 valid keys; actual last partial block makes this an upper envelope, route-dependent.')
    caps=[]
    for kk in [64,96,128,159]:
        payload=2*Llayers*HH*((N-Pn)*kk+Pn*(D+RR))
        rho_bytes=4*Llayers*HH*N
        basis_bytes=2*Llayers*HH*(D+RR)*kk
        anchor_flops=2*Llayers*HH*(N-Pn)*(D+RR)*kk
        hit_flops=2*Llayers*HH*(N-Pn)*kk*D
        decoder_flops=2*Llayers*HH*kk*(D+RR)*D
        original_history_flops=2*Llayers*HH*(N-Pn)*RR*D+Llayers*HH*(N-Pn)*D
        caps.append(dict(rank=kk,principal_state_GiB=(payload+rho_bytes)/2**30,
             per_layer_head_basis_GiB=basis_bytes/2**30,total_with_bases_GiB=(payload+rho_bytes+basis_bytes)/2**30,
             anchor_encoding_Tflop=anchor_flops/1e12,hit_late_readout_Tflop=hit_flops/1e12,
             decoder_preparation_Tflop=decoder_flops/1e12,original_hit_historical_terms_Tflop=original_history_flops/1e12,
             added_anchor_hit_arithmetic_Tflop=(anchor_flops+hit_flops+decoder_flops-original_history_flops)/1e12))
    out=dict(scope='CPU algebra with actual frozen CAL bases and synthetic Q/K/V and downstream matrices; zero model/GPU runs',
       frozen_basis=reports,partition=partition,probe_first_streaming=streaming,folded_high=fold,
       ridge_geometry=ridge,joint_state_compression=compression,production_costs=costs,joint_compression_capacity=caps)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    assert max(partition.values())<1e-11
    assert streaming['output_max_abs']<1e-10
    assert fold['real_max_abs']<1e-9 and fold['feature_rounding_defect_identity_max_abs']<1e-9
    assert ridge['pre_overwrite_rank']==n and ridge['post_overwrite_rank']==n-p
    assert compression['readout_identity_max_abs']<1e-10
if __name__=='__main__':main()
