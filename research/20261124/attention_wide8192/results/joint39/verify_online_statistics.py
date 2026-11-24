"""Check online attention statistics at Wan head/probe dimensions.

This is a SYNTHETIC ALGEBRA CHECK, not a real activation, model, quality or
latency experiment. It uses the archived projection bases and model dimensions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def stream(q, k, w, chunk=1024):
    m = np.full(len(q), -np.inf)
    ell = np.zeros(len(q))
    z = np.zeros((len(q), w.shape[1]))
    for start in range(0, len(k), chunk):
        score = q @ k[start:start+chunk].T / np.sqrt(k.shape[1])
        mb = score.max(axis=1)
        new_m = np.maximum(m, mb)
        old_scale = np.exp(m-new_m)
        exp_score = np.exp(score-new_m[:, None])
        ell = old_scale*ell + exp_score.sum(axis=1)
        z = old_scale[:, None]*z + exp_score @ w[start:start+chunk]
        m = new_m
    return z/ell[:, None], m+np.log(ell)


def direct(q, k, w):
    score = q @ k.T / np.sqrt(k.shape[1])
    m = score.max(axis=1)
    exp_score = np.exp(score-m[:, None])
    den = exp_score.sum(axis=1)
    prob = exp_score/den[:, None]
    return prob @ w, m+np.log(den), prob


def main(bases, output):
    n, p, d, s = 31668, 64, 128, 3968
    rng = np.random.default_rng(20261124)
    q, k, v = [rng.standard_normal(shape) for shape in ((p,d),(n,d),(n,d))]
    pv = np.load(bases/'P_v.npy', allow_pickle=False).astype(np.float64)[:, :8]
    pk = np.load(bases/'P_k.npy', allow_pickle=False).astype(np.float64)[:, :8]
    assert pv.shape == pk.shape == (128,8)
    full, lse, _ = direct(q,k,v)
    streamed, streamed_lse = stream(q,k,v,1031)
    vp, kp = v[:s]@pv, k[:s]@pk
    augmented = np.concatenate((v[:s],kp,vp*kp),axis=1)
    out, lsel = stream(q,k[:s],augmented,257)
    ref, ref_lse, prob = direct(q,k[:s],v[:s])
    mean_v, mean_k = prob@vp, prob@kp
    # Direct centered covariance per query; compare with one augmented sum.
    covariance = np.einsum('ij,ija,ija->ia',prob,
        vp[None,:,:]-mean_v[:,None,:],kp[None,:,:]-mean_k[:,None,:])
    covariance_from_sum = out[:,136:144] - (out[:,:128]@pv)*out[:,128:136]
    checks = {
        'full_output_max_abs':float(np.max(np.abs(full-streamed))),
        'full_lse_max_abs':float(np.max(np.abs(lse-streamed_lse))),
        'selected_output_max_abs':float(np.max(np.abs(ref-out[:,:128]))),
        'selected_lse_max_abs':float(np.max(np.abs(ref_lse-lsel))),
        'selected_key_mean_max_abs':float(np.max(np.abs(mean_k-out[:,128:136]))),
        'selected_covariance_max_abs':float(np.max(np.abs(covariance-covariance_from_sum))),
    }
    assert max(checks.values()) < 1e-11, checks
    result = {
        'evidence_type':'SYNTHETIC_ALGEBRA_CHECK_ONLY',
        'real_model_execution':False,'latency_measured':False,
        'dimensions':{'all_keys':n,'queries':p,'head_width':d,
                      'selected_keys':s,'augmented_value_width':144},
        'dtype':'NumPy FP64','seed':20261124,
        'input_semantics':'Synthetic Q/K/V; archived P_v and P_k; no model activations.',
        'basis_sha256':{name:hashlib.sha256((bases/name).read_bytes()).hexdigest()
                        for name in ('P_v.npy','P_k.npy')},
        'checks':checks,
        'conclusion':'The same online softmax reduction yields O and LSE; augmented values yield the requested projected key means and value/key covariance diagonal.',
        'execution_boundary':'Does not validate BF16 rounding, an installed fused kernel interface, throughput, memory peak, or video quality.'
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bases',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('online_statistics_algebra.json'))
    a=ap.parse_args()
    main(a.bases,a.output)
