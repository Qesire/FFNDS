"""Real archived Gram analysis, symbolic identities and labelled synthetic checks.

No model weights, GPU run, new video, or current six-source activation measurements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import sympy as s


def ln(x):
    c=x-x.mean(-1,keepdims=True)
    return c/np.sqrt((c*c).mean(-1,keepdims=True)+1e-6)


def softmax(x):
    z=np.exp(x-x.max(-1,keepdims=True))
    return z/z.sum(-1,keepdims=True)


def gelu(x):
    return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=here)
    p.add_argument('--output',type=Path,default=here/'joint_defect_verified.json')
    args=p.parse_args()
    rows=[]; hashes={}
    for source in ('0392','1013'):
        path=args.source_dir/f'RESULT_{source}.json'
        data=path.read_bytes();raw=json.loads(data)
        hashes[path.name]=hashlib.sha256(data).hexdigest()
        for background, bg in raw['results'].items():
            for expert, site in bg['experts'].items():
                for stage in ('block_exit','flow'):
                    info=site['branch_gram'][stage]
                    gram=np.asarray(info['gram'],dtype=np.float64)
                    assert info['terms']==['E_A','E_F','I_AF']
                    assert np.allclose(gram,gram.T)
                    assert np.linalg.eigvalsh(gram).min()>0
                    c=np.linalg.solve(gram[:2,:2],gram[:2,2])
                    perpendicular=float(gram[2,2]-gram[2,:2]@c)
                    total=float(gram.sum())
                    assert abs(total-info['B11_error'])<1e-9
                    assert perpendicular>=0
                    effective=np.array([1+c[0],1+c[1]])
                    reconstructed=float(effective@gram[:2,:2]@effective+perpendicular)
                    assert abs(reconstructed-total)<1e-9
                    rows.append({'source':f'vbi_{source}','background':background,'expert':expert,'stage':stage,
                        'projection_coefficients_I_on_EA_EF':c.tolist(),
                        'interaction_energy':float(gram[2,2]),'joint_error_energy':total,
                        'interaction_perpendicular_energy':perpendicular,
                        'interaction_perpendicular_fraction_of_interaction':perpendicular/gram[2,2],
                        'interaction_over_joint_error':float(gram[2,2]/total),
                        'interaction_perpendicular_over_joint_error':perpendicular/total,
                        'gram_min_eigenvalue':float(np.linalg.eigvalsh(gram).min()),
                        'closure_error':abs(reconstructed-total)})

    Fs,Fa,Gs,Ga,R=s.symbols('F_s F_a G_s G_a R',real=True)
    D=Fs-Gs; C=Fa-Ga
    exact_recovery=s.expand((Gs+C+R-Fs)-(R-(D-C)))==0
    Ta,Tb,FUa,FUb,GUb,g=s.symbols('T_a T_b F_Ua F_Ub G_Ub g',real=True)
    native=Ta+g*FUa; cheap=Tb+g*GUb
    pieces=(Ta-Tb)+g*(FUa-FUb)+g*(FUb-GUb)
    component_closure=s.expand(native-cheap-pieces)==0
    assert exact_recovery and component_closure

    # Synthetic finite nonlinear block. This only checks the algebra/observability
    # claims on a fully executable example, not an estimate for Wan activations.
    rng=np.random.default_rng(8121)
    n,d,m=48,16,40
    def weight(a,b):return rng.standard_normal((a,b))/np.sqrt(a)
    wq,wk,wv,wo=[weight(d,d) for _ in range(4)]
    wcq,wck,wcv,wco=[weight(d,d) for _ in range(4)]
    w1,w2=weight(d,m),weight(m,d)
    ctx=rng.standard_normal((11,d));ck=ctx@wck;cv=ctx@wcv
    selected=np.arange(0,n,4); hidden=np.arange(12)
    linear=.5*w1@w2
    def block(x,phase,cheap):
        u=ln(x)*(1+.07*phase)+.03*phase
        q,k,v=u@wq,u@wk,u@wv
        if cheap:k,v=k[selected],v[selected]
        a=softmax(q@k.T/np.sqrt(d))@v@wo
        r=x+(0.7+0.05*phase)*a
        cross=softmax((ln(r)@wcq)@ck.T/np.sqrt(d))@cv@wco
        t=r+cross
        z=(ln(t)*(1-.04*phase)+.02*phase)@w1
        if cheap:
            ff=(ln(t)*(1-.04*phase)+.02*phase)@linear+(gelu(z[:,hidden])-.5*z[:,hidden])@w2[hidden]
        else:ff=gelu(z)@w2
        return t+(0.6-.03*phase)*ff
    xa=rng.standard_normal((n,d));xs=xa+.35*rng.standard_normal((n,d))
    fa,ga=block(xa,0,False),block(xa,0,True)
    fs,gs=block(xs,1,False),block(xs,1,True)
    ca=fa-ga; ds=fs-gs; innovation=ds-ca
    exact=gs+ca+innovation
    probes=np.arange(0,n,6)
    overwritten_g=gs.copy();overwritten_g[probes]=fs[probes]
    shift=rng.standard_normal((n,1))
    native_equivariance=float(np.max(abs(block(xs+shift,1,False)-fs-shift)))
    cheap_equivariance=float(np.max(abs(block(xs+shift,1,True)-gs-shift)))
    assert np.max(abs(exact-fs))<1e-12
    assert native_equivariance<1e-12 and cheap_equivariance<1e-12
    synthetic={'scope':'Synthetic random weights and inputs, N48 d16 m40; not a real model quality result',
        'joint_anchor_return_max_abs':float(np.max(abs(ga+ca-fa))),
        'finite_innovation_reconstruction_max_abs':float(np.max(abs(exact-fs))),
        'pre_override_current_probe_defect_norm':float(np.linalg.norm((fs-gs)[probes])),
        'post_override_current_probe_defect_norm':float(np.linalg.norm((fs-overwritten_g)[probes])),
        'native_channel_shift_equivariance_max_abs':native_equivariance,
        'cheap_channel_shift_equivariance_max_abs':cheap_equivariance}
    N,H,dh,L=31668,40,128,40
    budget={'queries':N,'heads':H,'head_dim':dh,'model_dim':H*dh,'layers_in_resident_expert':L,
        'single_full_position_field_BF16_GiB':N*H*dh*L*2/2**30,
        'P64_full_block_teacher_F32_MiB_per_layer':64*H*dh*4/2**20,
        'H448_full_block_field_F32_MiB_per_layer':448*H*dh*4/2**20,
        'probe_query_fraction':64/N,
        'notes':['One block-defect field is sufficient for the uncorrected macro predictor, not for reusing the original FFN algorithm unchanged.',
                 'Anchor must evaluate cheap composite path on cheap-attention inputs; original native tail does not provide this value.',
                 'The four-step two-expert setting has one hit per anchor, so extra shadow construction has only one reuse to repay.']}
    out={'evidence_head':'873e1a879b91d544fdae11894d68bbc6acc854d1',
         'real_data_scope':'Historical terminal block39 four-arm H448 branch Gram; not six-source C activations',
         'real_source_sha256':hashes,'real_gram_projection':rows,
         'symbolic':{'joint_anchor_innovation_error_identity':exact_recovery,'joint_defect_three_part_identity':component_closure},
         'synthetic':synthetic,'real_shape_static_budget':budget}
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'symbolic':out['symbolic'],'synthetic':synthetic,
        'wide_flow':[r for r in rows if r['background']=='wide8192' and r['stage']=='flow'],
        'static_budget':budget},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
