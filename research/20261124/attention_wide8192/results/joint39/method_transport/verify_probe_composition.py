"""Verify shared-probe closure with a nonlinear tail and wide8192-form readout.

This is a deterministic synthetic semantic witness, not a Wan activation replay.
The deployed source is read separately; its 64-probe/16-mode dependency is kept.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def ln(x):
    c = x - x.mean(-1, keepdims=True)
    return c / np.sqrt(np.mean(c*c, axis=-1, keepdims=True) + 1e-5)


def gelu(x):
    return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x*x*x)))


def main():
    rng = np.random.default_rng(250911)
    n, d, m, k, modes, pn = 256, 32, 80, 24, 16, 64
    p = np.arange(0, n, n//pn)
    def normal(shape, scale=1.):
        return rng.standard_normal(shape)*scale
    x, attention = normal((n,d)), normal((n,d), .25)
    wo, cross, w1, w2 = [normal(shape, .1) for shape in
                         ((d,d),(d,d),(d,m),(m,d))]
    ga, gf = normal((d,),.3), normal((d,),.3)
    shift, scale = normal((d,),.1), 1+normal((d,),.1)
    def to_ffn(a):
        z = x + (a@wo)*ga
        z = z + np.tanh(ln(z)@cross)
        u = ln(z)*scale+shift
        return z,u
    def native(u):
        return gelu(u@w1)@w2
    ua = normal((n,d), .7)
    ya, z0 = native(ua), ua@w1[:,:k]
    a0 = w1[:,k:]@w2[k:]*.5
    zmap, jmap, v = normal((k,modes),.07),normal((d,modes),.05),normal((modes,d),.1)
    def adaptive_ffn(u):
        du = u-ua
        z = u@w1[:,:k]
        dh = gelu(z)-gelu(z0)
        basis = ya+du@a0+dh@w2[:k]
        q = dh@zmap-du@jmap
        features = np.tanh(z-z0)
        bp, qp, xp = basis[p],q[p],features[p]
        observed = native(u[p])
        target = observed-bp
        system = xp@xp.T/k+.5*np.eye(pn)
        iq = np.linalg.solve(system,qp)
        gain = np.linalg.solve((qp.T@iq)*(v@v.T),np.sum(iq*(target@v.T),axis=0))
        coeff = np.linalg.solve(system,target-(qp*gain)@v)
        out = basis+(q*gain)@v+.5*(features@xp.T/k)@coeff
        out[p] = observed
        return out, {'gain':gain,'coeff':coeff,'xp':xp,'qp':qp,'observed':observed}
    def block(a):
        z,u = to_ffn(a)
        f,readout = adaptive_ffn(u)
        return z+f*gf,u,readout
    err = normal((n,d),.18)
    unprotected = attention+err
    protected = unprotected.copy()
    protected[p] = attention[p]
    y0,u0,r0 = block(attention)
    yp,up,rp = block(protected)
    yu,uu,ru = block(unprotected)
    diffs = {key:float(np.max(np.abs(rp[key]-r0[key]))) for key in r0}
    assert max(diffs.values()) < 1e-12
    assert np.max(np.abs(up[p]-u0[p])) < 1e-12
    assert np.max(np.abs(yp[p]-y0[p])) < 1e-12
    assert np.linalg.norm(yp-y0)>1e-3
    assert np.linalg.norm(ru['gain']-r0['gain'])>1e-3
    here=Path(__file__).parent
    sources={}
    for name in ('fourstep_backend.py','fourstep_hit.py','attention_online_structured.py'):
        path=here/'source'/name
        raw=path.read_bytes()
        sources[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    out={
        'scope':'Synthetic nonlinear composition with the deployed readout dependency; no Wan weights/activations or GPU execution',
        'seed':250911,
        'shape':dict(tokens=n,channels=d,hidden=m,selected=k,shared_modes=modes,probes=pn),
        'assumptions':['Same current block input, time/text conditions and FFN cache',
                       'Same probe positions in attention and FFN',
                       'Attention probe values exactly match the chosen native probe chain',
                       'Intervening cross/LN/projections/gates are rowwise in the video-query axis',
                       'FFN global readout depends only on current probe inputs, probe caches and fixed assets'],
        'protected_ffn_input_probe_max_error':float(np.max(np.abs(up[p]-u0[p]))),
        'protected_readout_max_differences':diffs,
        'protected_block_probe_max_error':float(np.max(np.abs(yp[p]-y0[p]))),
        'protected_nonprobe_block_error_norm':float(np.linalg.norm(yp-y0)),
        'unprotected_gain_difference_norm':float(np.linalg.norm(ru['gain']-r0['gain'])),
        'unprotected_coeff_difference_norm':float(np.linalg.norm(ru['coeff']-r0['coeff'])),
        'deployed_source_evidence_commit':'7c8196fa5380dbe67dd9772f942553085fde55b3',
        'deployed_sources':sources,
        'source_binding':'tools/ffnds_fourstep/fourstep_backend.py maps wide8192 to fourstep_hit.make_hit(8192,reuse_base=True)',
        'source_audit':'fourstep_hit computes gain/coeff/xp only from operands(probes), native FFN(u[probes]) and fixed parameters; nonprobe operands are computed afterwards'
    }
    path=here/'probe_composition_verified.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
