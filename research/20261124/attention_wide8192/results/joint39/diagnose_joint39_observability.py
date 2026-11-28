"""Diagnose the fixed-P64 function space in an existing D176 CPU replay.

No model execution, new probes, tuning, or H-label candidate selection occurs.
The H-label fits are diagnostic lower bounds only. Run next to the archived
attention_joint39_d176.py and its dependencies, using the SAME five positional
arguments as that producer: reduced fit export index out.

For local algebra verification without torch/assets:
  python diagnose_joint39_observability.py --self-check

The ordinary path reuses the producer's exact asset loading, fingerprint checks,
P-only normalization and design construction. It adds diagnostics to its output;
it never edits the producer or changes its P64 predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def orth(a, reference_scale=None):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    scale = s[0] if len(s) else 0.0
    if reference_scale is not None:
        scale=max(scale,reference_scale)
    tol = np.finfo(np.float64).eps * max(a.shape) * scale
    rank = int(np.count_nonzero(s > tol))
    return u[:, :rank], s, vt, rank, float(tol)


def energy(a):
    return float(np.sum(np.square(a), dtype=np.float64))


def diagnose(d, base, truth, p, h, ridge=0.1):
    """All matrices use query rows; target columns are the 64 flow channels."""
    d, base, truth = [np.asarray(v, dtype=np.float64) for v in (d, base, truth)]
    p, h = [np.asarray(v, dtype=np.int64) for v in (p, h)]
    assert d.ndim == 2 and base.shape == truth.shape and base.shape[0] == len(d)
    assert not set(p).intersection(h) and len(set(p)) == len(p)
    dp, dh = d[p], d[h]
    rp, rh = (truth-base)[p], (truth-base)[h]
    m = dh @ dp.T
    q_full, s_h, _, rank_h, tol_h = orth(dh)
    q_pspan, s_m, _, rank_m, tol_m = orth(m)
    fit_full = q_full @ (q_full.T @ rh)
    fit_pspan = q_pspan @ (q_pspan.T @ rh)
    e_full = energy(rh-fit_full)
    e_pspan = energy(rh-fit_pspan)

    gram = dp @ dp.T
    reg = len(p) * ridge
    alpha = np.linalg.solve(gram + reg*np.eye(len(p)), rp)
    fit_actual = m @ alpha
    actual64 = energy(rh-fit_actual)
    # Match producer: whole prediction rounded to FP32 before FP64 scoring.
    prediction32 = (base[h] + fit_actual).astype(np.float32).astype(np.float64)
    actual32 = energy(prediction32-truth[h])

    # Orthogonal nested-space decomposition. These terms are nonnegative;
    # the last term jointly includes P-label estimation and regularization.
    lost_space = energy(fit_full-fit_pspan)
    coefficient_gap = energy(fit_pspan-fit_actual)
    residual_check = abs(actual64 - e_full - lost_space - coefficient_gap)
    nested_check = energy(q_pspan-q_full@(q_full.T@q_pspan))
    tol_energy = 1e-9*max(1.0, actual64, energy(rh))
    assert residual_check <= tol_energy
    assert nested_check <= 1e-12*max(1, rank_m)

    # Use an orthonormal function basis on P union H, so a coefficient-null
    # direction means a genuinely different function, not a redundant column.
    stacked = np.concatenate((dp,dh), axis=0)
    q_all, s_all, _, rank_all, tol_all = orth(stacked)
    up = q_all[:len(p)]
    _, s_restrict, _, rank_restrict, tol_restrict = orth(up)
    _, s_p, vt_p, rank_p, tol_p = orth(dp)
    row_projector = vt_p[:rank_p].T @ vt_p[:rank_p]
    null_projector = np.eye(d.shape[1])-row_projector
    null_image = dh @ null_projector
    _, s_null, _, rank_null, tol_null = orth(null_image,reference_scale=float(s_h[0]))
    gram_eigenvalues = np.linalg.eigvalsh(gram)
    gains = gram_eigenvalues/(gram_eigenvalues+reg)
    transfer = np.linalg.solve(gram+reg*np.eye(len(p)), m.T).T
    transfer_s = np.linalg.svd(transfer, compute_uv=False)
    e_nonzero = s_restrict[s_restrict > tol_restrict]
    return {
        "scope": "H448 diagnostics; no H labels enter the original P64 predictor",
        "dimensions": {"p":len(p),"h":len(h),"columns":d.shape[1],"target_channels":truth.shape[1]},
        "ranks": {"D_P":rank_p,"D_H":rank_h,"D_H_D_P_T":rank_m,
                  "stacked_P_H":rank_all,"sample_restriction":rank_restrict,
                  "sample_invisible_function_dimension":rank_all-rank_restrict,
                  "D_H_null_D_P":rank_null},
        "point_error": {"base":energy(rh),"full_H_OLS":e_full,
                        "P_span_H_OLS":e_pspan,"actual_P_ridge_FP64":actual64,
                        "actual_P_ridge_FP32_prediction":actual32},
        "orthogonal_error_budget_FP64": {
            "outside_full_dictionary":e_full,
            "full_dictionary_to_P_representer_span":lost_space,
            "P_span_coefficient_gap":coefficient_gap,
            "sum_check_absolute":residual_check,
            "rounding_delta_in_energy":actual32-actual64,
            "interpretation":"Coefficient gap combines P supervision, regularization and fit; it is not pure sampling variance."},
        "stability": {"ridge":ridge,"n_lambda":reg,
                      "regularized_probe_gram_condition":float((gram_eigenvalues[-1]+reg)/(gram_eigenvalues[0]+reg)),
                      "effective_probe_degrees_of_freedom":float(np.sum(gains)),
                      "P_to_H_ridge_operator_norm":float(transfer_s[0]) if len(transfer_s) else 0.0,
                      "sample_restriction_smallest_nonzero_singular_value":float(e_nonzero[-1]) if len(e_nonzero) else None,
                      "coefficient_null_probe_energy":energy(dp@null_projector),
                      "coefficient_null_H_operator_norm":float(s_null[0]) if len(s_null) else 0.0},
        "singular_values": {"D_P":s_p.tolist(),"D_H":s_h.tolist(),
                            "D_H_D_P_T":s_m.tolist(),"P_restriction_of_orthonormal_function_basis":s_restrict.tolist()},
        "rank_tolerances": {"D_P":tol_p,"D_H":tol_h,"D_H_D_P_T":tol_m,
                            "stacked_P_H":tol_all,"sample_restriction":tol_restrict,"D_H_null_D_P":tol_null},
    }


def algebra_check():
    """Small deterministic counterexample, not a measurement on Wan."""
    # P observes the first two directions; H separately exposes all four.
    d=np.array([[1,0,0,0],[0,1,0,0],[1,0,0,0],[0,1,0,0],
                [0,0,1,0],[0,0,0,1]], dtype=np.float64)
    truth=np.array([[1],[2],[1],[2],[3],[4]], dtype=np.float64)
    out=diagnose(d,np.zeros_like(truth),truth,np.array([0,1]),np.array([2,3,4,5]))
    assert out["ranks"]["sample_invisible_function_dimension"]==2
    assert abs(out["point_error"]["full_H_OLS"])<1e-12
    assert abs(out["point_error"]["P_span_H_OLS"]-25)<1e-12
    # In the rank<=P case with full column rank at P, the two function spans agree.
    reduced=diagnose(d[:,:2],np.zeros_like(truth),truth,np.array([0,1]),np.array([2,3,4,5]))
    assert abs(reduced["point_error"]["full_H_OLS"]-reduced["point_error"]["P_span_H_OLS"])<1e-12
    print(json.dumps({"evidence_type":"synthetic_algebra_check_only",
                      "underdetermined":out,"identified_function_space":reduced},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check",action="store_true")
    ap.add_argument("--implementation-dir",type=Path,default=Path(__file__).resolve().parent)
    for name in ("reduced","fit","export","index","out"):
        ap.add_argument(name,type=Path,nargs="?")
    args=ap.parse_args()
    if args.self_check:
        algebra_check()
        return
    paths=[getattr(args,k) for k in ("reduced","fit","export","index","out")]
    if any(p is None for p in paths):
        ap.error("Supply the same five positional paths as attention_joint39_d176.py.")
    sys.path.insert(0,str(args.implementation_dir.resolve()))
    original=importlib.import_module("attention_joint39_d176")
    original_metrics=original.four_arm_metrics
    def with_diagnostic(d,base,truth,p,h,edges):
        result=original_metrics(d,base,truth,p,h,edges)
        as_array=lambda t:t.detach().cpu().double().numpy()
        for key in original.ORDER:
            report=diagnose(as_array(d[key]),as_array(base),as_array(truth),
                            p.cpu().numpy(),h.cpu().numpy())
            a=result[key]["P64_fit"]["point"]
            b=report["point_error"]["actual_P_ridge_FP32_prediction"]
            assert abs(a-b)<=1e-5*max(1.0,a), (key,a,b)
            report["producer_point_energy_check_relative"]=abs(a-b)/max(1.0,a)
            result[key]["information_diagnostic"]=report
        return result
    original.four_arm_metrics=with_diagnostic
    original.main(*(p.resolve() for p in paths))
    output=paths[-1].resolve()
    result=json.loads(output.read_text())
    result["diagnostic_producer_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["diagnostic_status"]="complete; all H-label estimates diagnostic only"
    result["candidate_predictions_changed"]=False
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
