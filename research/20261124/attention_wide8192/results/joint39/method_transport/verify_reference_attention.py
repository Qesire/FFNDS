"""Execute fixed mathematical witnesses for the CPU method reference.

The projection bytes are archived FFNDS assets. All Q/K/V arrays in this file
are explicitly synthetic. No new Wan inference, video or GPU timing is run.
"""
from dataclasses import replace
from pathlib import Path
import argparse,hashlib,io,json,zipfile
import numpy as np
from reference_attention import (Config,build_anchor,recover_hit,full_attention,
    group_labels,grouped_attention,defect_projection)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_assets(basis_dir,replay_zip):
    inventory=[]
    if basis_dir is not None:
        blobs={n:(basis_dir/n).read_bytes() for n in ('P_k.npy','P_v.npy')}
        for p in sorted(basis_dir.parent.parent.rglob('*.npz')):
            with np.load(p,allow_pickle=False) as z:
                inventory.append({'file':p.name,'arrays':{k:list(z[k].shape) for k in z.files}})
        source={'kind':'basis_directory','basis_dir':str(basis_dir)}
    else:
        with zipfile.ZipFile(replay_zip) as z:
            blobs={n:z.read('plan_assets/bases/'+n) for n in ('P_k.npy','P_v.npy')}
            entries=z.namelist()
            for name in entries:
                if name.endswith('.npz'):
                    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as f:
                        inventory.append({'file':name,'arrays':{k:list(f[k].shape) for k in f.files}})
            source={'kind':'replay_archive','entries':entries,'sha256':sha(replay_zip.read_bytes())}
    pk,pv=[np.load(io.BytesIO(blobs[n]),allow_pickle=False).astype(np.float64) for n in ('P_k.npy','P_v.npy')]
    source['basis_sha256']={n:sha(v) for n,v in blobs.items()}
    source['basis_shapes']={'P_k':list(pk.shape),'P_v':list(pv.shape)}
    source['calibration_inventory']=inventory
    source['available_activation_arrays']=[]
    source['activation_scope']='Read archive/directory contains projections, calibration transforms and summaries, not full anchor/hit Q/K/V.'
    return pk,pv,source


class ProbeTeacher:
    def __init__(self):
        self.calls=[]
    def __call__(self,q,k,v):
        self.calls.append({'heads':q.shape[0],'query_rows':q.shape[1],'full_keys':k.shape[1]})
        return full_attention(q,k,v)[0]


def maxabs(x):
    return float(np.max(np.abs(x)))


def support(H,I,J,config):
    qb=(I+config.query_block-1)//config.query_block
    kb=(J+config.key_block-1)//config.key_block
    out=np.zeros((H,qb,kb),dtype=bool)
    for h in range(H):
        for b in range(qb): out[h,b,(b+h)%kb]=True
    return out


def verify(pk,pv):
    rng=np.random.default_rng(20261125)
    H,N,D=2,190,128
    cfg=Config(key_block=64,query_block=64,query_chunk=32,coarse_chunk=2)
    P=np.linspace(0,N-1,64,dtype=np.int64)
    hold=np.ones(N,dtype=bool);hold[P]=False
    sel=support(H,N,N,cfg)
    q=rng.normal(size=(H,N,D))*.8
    k=rng.normal(size=(H,N,D))
    k[:,0:64]=0. # A genuinely empty sign subgroup.
    v=.3*k+rng.normal(size=(H,N,D))*.7
    states={}
    anchor_results={}
    for mode in ('raw','channel_normalized'):
        st,base,native=build_anchor(q,k,v,sel,pk,pv,cfg,cache_mode=mode)
        teacher=ProbeTeacher()
        hit=recover_hit(q,k,v,pk,pv,P,st,teacher=teacher)
        states[mode]=st
        anchor_results[mode]={'max_abs_output_error':maxabs(hit.output-native),
            'max_abs_innovation':maxabs(hit.probe_residual),
            'teacher_calls':teacher.calls}
        assert maxabs(hit.output-native)<2e-12
        assert teacher.calls==[{'heads':H,'query_rows':64,'full_keys':N}]
    st=states['channel_normalized']

    # Fixed Q/K and per-head/channel positive value affine evolution.
    gains=np.exp(rng.normal(size=(H,1,D))*.45)
    bias=rng.normal(size=(H,1,D))*.3
    v_gain=v*gains+bias
    native_gain=full_attention(q,k,v_gain)[0]
    raw_gain=recover_hit(q,k,v_gain,pk,pv,P,states['raw'])
    norm_gain=recover_hit(q,k,v_gain,pk,pv,P,st)
    affine_class={
        'raw_unfitted_base_holdout_l2':float(np.linalg.norm((raw_gain.base.output+raw_gain.anchor_term-native_gain)[:,hold])),
        'raw_recovered_holdout_l2':float(np.linalg.norm((raw_gain.output-native_gain)[:,hold])),
        'channel_normalized_unfitted_base_max_abs_error':maxabs(norm_gain.base.output+norm_gain.anchor_term-native_gain),
        'channel_normalized_recovered_max_abs_error':maxabs(norm_gain.output-native_gain),
        'gain_range':[float(gains.min()),float(gains.max())],
        'scope':'Synthetic exact action class, fixed Q/K/support/groups, strictly positive diagonal gains, inactive scale floor.'}
    assert affine_class['raw_unfitted_base_holdout_l2']>1e-2
    assert affine_class['channel_normalized_recovered_max_abs_error']<3e-12

    # A general finite next state: arbitrary Q/K/V changes, fixed stored groups.
    q1=q+.3*rng.normal(size=q.shape)
    k1=k+.5*rng.normal(size=k.shape)
    v1=v+.4*rng.normal(size=v.shape)
    labels_before=st.labels.copy();support_before=st.selected_blocks.copy()
    t=ProbeTeacher();hit=recover_hit(q1,k1,v1,pk,pv,P,st,teacher=t)
    native1=full_attention(q1,k1,v1)[0]
    regrouped=group_labels(k1,pk[:,0],cfg.key_block)
    finite={'changed_memberships_if_regrouped':int(np.count_nonzero(regrouped!=st.labels)),
        'stored_labels_unchanged':bool(np.array_equal(labels_before,st.labels)),
        'stored_support_unchanged':bool(np.array_equal(support_before,st.selected_blocks)),
        'current_probe_innovation_l2':float(np.linalg.norm(hit.probe_residual)),
        'teacher_calls':t.calls,
        'final_P_exact_max_abs_error':maxabs(hit.output[:,P]-hit.teacher),
        'feature_shape':list(hit.features.shape),
        'ridge_alpha':len(P)*cfg.ridge_per_probe,
        'empty_group_count':int(np.count_nonzero(hit.base.groups.counts==0)),
        'last_valid_keys':N%cfg.key_block}
    assert finite['changed_memberships_if_regrouped']>0
    assert finite['stored_labels_unchanged'] and finite['stored_support_unchanged']
    assert finite['current_probe_innovation_l2']>1e-2
    assert finite['final_P_exact_max_abs_error']==0.

    # Isolate ONLY historical transport: both paths use the very same current
    # normalized features, P teacher, scale, ridge and final native P overwrite.
    # The override is in output units and thus supplies raw E_a without changing
    # any current-feature semantics. This is distinct from the raw-mode control.
    raw_same_gain=recover_hit(q,k,v_gain,pk,pv,P,st,
        transported_anchor_defect=states['raw'].defect)
    raw_same_finite=recover_hit(q1,k1,v1,pk,pv,P,st,
        transported_anchor_defect=states['raw'].defect)
    transport_identity=[]
    for normalized,untransported in ((norm_gain,raw_same_gain),(hit,raw_same_finite)):
        assert np.array_equal(normalized.features,untransported.features)
        assert np.array_equal(normalized.teacher,untransported.teacher)
        for h in range(H):
            x=normalized.features[h];xp=x[P]
            kr=x@np.linalg.solve(xp.T@xp+len(P)*cfg.ridge_per_probe*np.eye(x.shape[-1]),xp.T)
            db=normalized.anchor_term[h]-untransported.anchor_term[h]
            pred=db-kr@db[P];pred[P]=0.
            transport_identity.append(maxabs(normalized.output[h]-untransported.output[h]-pred))
    isolated_transport={
        'features_and_teacher_bitwise_equal':True,
        'final_output_difference_identity_max_abs':max(transport_identity),
        'positive_diagonal_same_features_untransported_holdout_l2':float(np.linalg.norm((raw_same_gain.output-native_gain)[:,hold])),
        'positive_diagonal_same_features_transported_holdout_l2':float(np.linalg.norm((norm_gain.output-native_gain)[:,hold])),
        'finite_same_features_untransported_holdout_l2':float(np.linalg.norm((raw_same_finite.output-native1)[:,hold])),
        'finite_same_features_transported_holdout_l2':float(np.linalg.norm((hit.output-native1)[:,hold])),
        'scope':'Only anchor output-unit field changes. Identity M_H(I-KP)(C_a D_s-E_a) holds for fixed current features; this synthetic witness is not a universal improvement claim.'}
    assert isolated_transport['final_output_difference_identity_max_abs']<3e-12
    assert isolated_transport['positive_diagonal_same_features_untransported_holdout_l2']>1e-2
    assert isolated_transport['positive_diagonal_same_features_transported_holdout_l2']<3e-12

    # Whole method equivariance under the SAME positive diagonal affine change
    # to anchor and hit V, including a nonzero finite innovation and ridge fit.
    st_aff,base_aff,_=build_anchor(q,k,v*gains+bias,sel,pk,pv,cfg,cache_mode='channel_normalized')
    hit_aff=recover_hit(q1,k1,v1*gains+bias,pk,pv,P,st_aff)
    equivariance={'anchor_normalized_cache_max_abs_difference':maxabs(st_aff.defect-st.defect),
        'current_features_max_abs_difference':maxabs(hit_aff.features-hit.features),
        'fitted_coefficients_max_abs_difference':maxabs(hit_aff.coefficients-hit.coefficients),
        'full_output_affine_error':maxabs(hit_aff.output-(hit.output*gains+bias)),
        'normalized_target_l2':float(np.linalg.norm(hit.probe_residual/hit.value_scale[:,None,:]))}
    assert max(equivariance[k] for k in equivariance if k.endswith(('difference','error')))<4e-12

    # Anchor and hit may each use a DIFFERENT positive diagonal affine value
    # coordinate change. Q/K at each corresponding call stay fixed, while
    # anchor Q/K and hit Q/K are allowed to differ as in this finite example.
    frame_rng=np.random.default_rng(20261126)
    hit_gains=np.exp(frame_rng.normal(size=(H,1,D))*.4)
    hit_bias=frame_rng.normal(size=(H,1,D))*.5
    independent_hit=recover_hit(q1,k1,v1*hit_gains+hit_bias,pk,pv,P,st_aff)
    independent_equivariance={
        'anchor_and_hit_gain_difference_l2':float(np.linalg.norm(gains-hit_gains)),
        'anchor_and_hit_shift_difference_l2':float(np.linalg.norm(bias-hit_bias)),
        'current_features_max_abs_difference':maxabs(independent_hit.features-hit.features),
        'fitted_coefficients_max_abs_difference':maxabs(independent_hit.coefficients-hit.coefficients),
        'normalized_innovation_max_abs_difference':maxabs(independent_hit.probe_residual/independent_hit.value_scale[:,None,:]-hit.probe_residual/hit.value_scale[:,None,:]),
        'full_output_affine_error':maxabs(independent_hit.output-(hit.output*hit_gains+hit_bias)),
        'scope':'Different positive diagonal gains and shifts at anchor and hit; Q/K unchanged by each frame transformation but differ across the two calls; floor inactive.'}
    assert independent_equivariance['anchor_and_hit_gain_difference_l2']>1.
    assert max(independent_equivariance[k] for k in independent_equivariance if k.endswith(('difference','error')))<4e-12

    # Current constant channels require zero defect correction, including when
    # their anchor fields were nonzero. Test some channels and then all columns.
    vpartial=v1.copy();vpartial[:,:,0:7]=bias[:,:,0:7]
    partial=recover_hit(q1,k1,vpartial,pk,pv,P,st)
    allconstant=np.broadcast_to(bias,v1.shape).copy()
    tc=ProbeTeacher();constant=recover_hit(q1,k1,allconstant,pk,pv,P,st,teacher=tc)
    constant_results={'partial_constant_columns_max_abs_error':maxabs(partial.output[:,:,0:7]-bias[:,:,0:7]),
        'partial_constant_columns_coefficients_max_abs':maxabs(partial.coefficients[:,:,0:7]),
        'all_constant_max_abs_error':maxabs(constant.output-bias),
        'all_constant_native_teacher_calls':tc.calls,
        'all_constant_anchor_term_max_abs':maxabs(constant.anchor_term)}
    assert constant_results['partial_constant_columns_max_abs_error']<2e-12
    assert constant_results['partial_constant_columns_coefficients_max_abs']==0.
    assert constant_results['all_constant_max_abs_error']<2e-12 and not tc.calls
    assert constant_results['all_constant_anchor_term_max_abs']==0.

    # Constant anchor columns have an exact zero normalized defect; the next
    # call may develop variance and fits its now nonzero current defect.
    st0,_,_=build_anchor(q,k,allconstant,sel,pk,pv,cfg,cache_mode='channel_normalized')
    assert maxabs(st0.defect)==0.
    hit0=recover_hit(q1,k1,v1,pk,pv,P,st0)
    constant_results['constant_anchor_cache_max_abs']=maxabs(st0.defect)
    constant_results['constant_anchor_to_variable_current_is_finite']=bool(np.isfinite(hit0.output).all())

    # The precise class represented by grouped summaries: K constant within each
    # frozen group, values arbitrary. This tests counts and incomplete blocks.
    direction=pk[:,0]/np.linalg.norm(pk[:,0])
    ke=np.empty_like(k)
    for b,start in enumerate(range(0,N,cfg.key_block)):
        end=min(start+cfg.key_block,N)
        signs=np.where(np.arange(end-start)%2==0,1.,-1.)
        ke[:,start:end]=(b*.2+signs[None,:,None])*direction
    ve=rng.normal(size=v.shape)
    ste,_,_=build_anchor(q,ke,ve,sel,pk,pv,cfg,cache_mode='channel_normalized')
    ke1=ke.copy()
    for h in range(H):
        for b,start in enumerate(range(0,N,cfg.key_block)):
            for side in (0,1):
                ids=np.arange(start,min(start+cfg.key_block,N))
                ids=ids[ste.labels[h,ids]==side]
                ke1[h,ids]=rng.normal(size=D)*.25+(2*side-1)*direction
    ve1=ve+.2*rng.normal(size=ve.shape)
    he=recover_hit(q1,ke1,ve1,pk,pv,P,ste)
    nativee=full_attention(q1,ke1,ve1)[0]
    group_exact={'anchor_cache_max_abs':maxabs(ste.defect),
        'current_base_max_abs_error':maxabs(he.base.output-nativee),
        'current_method_max_abs_error':maxabs(he.output-nativee)}
    assert max(group_exact.values())<3e-12

    # Opposing omitted K/V: the added current information resolves B collisions.
    vd=pv[:,0]/np.linalg.norm(pv[:,0])
    kk=np.concatenate([np.zeros((128,D)),4*direction[None],-4*direction[None]])[None]
    vv=np.concatenate([np.zeros((128,D)),vd[None],-vd[None]])[None]
    ts=np.array([-3.,-1.,0.,1.,3.])
    qq=(ts[:,None]*np.sqrt(D)/4*direction)[None]
    cs=Config(); ss=np.array([[[True,False]]])
    cc=group_labels(kk,direction,128)
    gb=grouped_attention(qq,kk,vv,ss,cc,cs)
    exact=full_attention(qq,kk,vv)[0]
    closed=(2*np.sinh(ts)/(128+2*np.cosh(ts)))[:,None]*vd
    collision={'selected_max_abs':maxabs(gb.selected),
        'grouped_max_abs_error':maxabs(gb.output-exact),
        'closed_form_max_abs_error':maxabs(exact[0]-closed),
        'full_response_coefficients':(exact[0]@vd).tolist()}
    assert collision['selected_max_abs']==0.
    assert max(collision['grouped_max_abs_error'],collision['closed_form_max_abs_error'])<2e-12

    # General value-linear transformation is an exact property of the CURRENT
    # base; the full fitted candidate only claims the diagonal-positive family.
    permutation=rng.permutation(D)
    mapped=grouped_attention(q1,k1,v1[:,:,permutation]*1.7+bias,st.selected_blocks,st.labels,cfg)
    base_equiv=maxabs(mapped.output-(hit.base.output[:,:,permutation]*1.7+bias))
    assert base_equiv<3e-12

    # Verify the finite residual operator at fixed current features, including
    # a cache perturbation and exact original P-cache preservation.
    raw=states['raw'];rawhit=recover_hit(q1,k1,v1,pk,pv,P,raw)
    eps=.003*rng.normal(size=raw.defect.shape)
    perturbed=recover_hit(q1,k1,v1,pk,pv,P,replace(raw,defect=raw.defect+eps))
    finite_identity=[];cache_identity=[]
    for h in range(H):
        x=rawhit.features[h];xp=x[P]
        kr=x@np.linalg.solve(xp.T@xp+64*.1*np.eye(x.shape[-1]),xp.T)
        delta=native1[h]-rawhit.base.output[h]-raw.defect[h]
        pred=kr@(delta-eps[h])[P]-(delta-eps[h]);pred[P]=0.
        finite_identity.append(maxabs((perturbed.output[h]-native1[h])-pred))
        predcache=eps[h]-kr@eps[h,P];predcache[P]=0.
        cache_identity.append(maxabs((perturbed.output[h]-rawhit.output[h])-predcache))
    protected=eps.copy();protected[:,P]=0.
    protected_hit=recover_hit(q1,k1,v1,pk,pv,P,replace(raw,defect=raw.defect+protected))
    operator={'finite_error_identity_max_abs':max(finite_identity),
        'cache_error_identity_max_abs':max(cache_identity),
        'P_protected_coefficients_bitwise_equal':bool(np.array_equal(protected_hit.coefficients,rawhit.coefficients)),
        'P_protected_nonP_error_identity_max_abs':maxabs((protected_hit.output-rawhit.output-protected)[:,hold]),
        'pre_overwrite_probe_residual_norm':float(np.linalg.norm(rawhit.probe_residual)),
        'residual_after_overwriting_base_with_teacher':0.}
    assert max(operator['finite_error_identity_max_abs'],operator['cache_error_identity_max_abs'],operator['P_protected_nonP_error_identity_max_abs'])<3e-12
    assert operator['P_protected_coefficients_bitwise_equal']

    # The production candidate's state error is in normalized coordinates;
    # decode it with current sigma after the same recovery operator.
    epsn=.002*rng.normal(size=st.defect.shape)
    norm_perturbed=recover_hit(q1,k1,v1,pk,pv,P,replace(st,defect=st.defect+epsn))
    normalized_identity=[]
    for h in range(H):
        x=hit.features[h];xp=x[P]
        kr=x@np.linalg.solve(xp.T@xp+64*.1*np.eye(x.shape[-1]),xp.T)
        dc=(native1[h]-hit.base.output[h])/hit.value_scale[h]-st.defect[h]
        pred=(kr@(dc-epsn[h])[P]-(dc-epsn[h]))*hit.value_scale[h]
        pred[P]=0.
        normalized_identity.append(maxabs(norm_perturbed.output[h]-native1[h]-pred))
    operator['normalized_cache_finite_error_identity_max_abs']=max(normalized_identity)
    assert operator['normalized_cache_finite_error_identity_max_abs']<3e-12

    # Streaming merge remains finite at logits that would overflow naive exp.
    extreme=grouped_attention(q1*1000,k1,v1,st.selected_blocks,st.labels,cfg)
    assert np.isfinite(extreme.output).all() and np.isfinite(extreme.lse).all()

    # No-anchor entry uses only current state plus P teacher.
    noanchor_teacher=ProbeTeacher()
    noanchor=recover_hit(q1,k1,v1,pk,pv,P,selected_blocks=sel,config=cfg,teacher=noanchor_teacher)
    assert np.isfinite(noanchor.output).all() and maxabs(noanchor.anchor_term)==0.
    return {'synthetic_shape':{'heads':H,'queries':N,'keys':N,'channels':D,'probes':64,'key_block':64},
        'precision':'float64; no BF16 cache/kernel equivalence claim',
        'anchor_return':anchor_results,'positive_diagonal_value_evolution':affine_class,
        'finite_hit_and_teacher':finite,'whole_method_positive_diagonal_affine_equivariance':equivariance,
        'independent_anchor_hit_affine_frames':independent_equivariance,
        'isolated_historical_transport':isolated_transport,
        'constant_channels':constant_results,'groupwise_constant_keys':group_exact,
        'selected_output_collision':collision,'current_base_general_value_map_error':base_equiv,
        'fixed_feature_operator':operator,
        'extreme_logits_output_and_lse_finite':True,
        'no_anchor_teacher_calls':noanchor_teacher.calls,
        'optional_anchor_dictionary_shape':list(defect_projection(st,pv[:,:15]).shape)}


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basis-dir',type=Path)
    p.add_argument('--replay-zip',type=Path,default=here.parent.parent/'replay_evidence.zip')
    p.add_argument('--output',type=Path,default=here/'reference_attention_verified.json')
    args=p.parse_args()
    pk,pv,assets=load_assets(args.basis_dir,args.replay_zip)
    checks=verify(pk,pv)
    result={'status':'CPU_MATHEMATICAL_REFERENCE_EXECUTED',
        'evidence_commit':'7c8196fa5380dbe67dd9772f942553085fde55b3',
        'scope':'Real archived projection bytes; explicitly synthetic Q/K/V; no new model inference, video, GPU execution, speed or Wan quality claim.',
        'implementation_sha256':sha((here/'reference_attention.py').read_bytes()),
        'assets':assets,'checks':checks,
        'binding':{'native_teacher':'CPU reference only; production q64 FlashAttention binding remains unexecuted',
                   'cache':'float64 here; proposed production field BF16 needs separate numeric acceptance',
                   'candidate_status':'Complete method semantics and CPU reference; actual-model replay requires matched effective anchor/hit Q,K,V, frozen selected support and probe indices, or a separately specified sufficient precomputed-statistic bundle'}}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
