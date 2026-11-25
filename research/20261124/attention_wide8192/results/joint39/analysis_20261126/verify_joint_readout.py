"""Recompute actual fixed-trajectory energies and verify task-readout algebra.

Real evidence: the archived Wan receipt, not newly executed model inference.
All arrays generated below are explicit mathematical witnesses, not Wan states.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "method_transport"
COMMIT = "253d2b72569289469ef191086e81643c90c37cfb"


def geometric(values):
    return float(np.exp(np.log(values).mean()))


def run():
    receipt_path = SOURCE / "wan_runtime_receipt_198062.json"
    summary = json.loads((SOURCE / "wan_runtime_summary_198062.json").read_text())
    receipt = json.loads(receipt_path.read_text())
    observed = receipt["replay"]["observations"]
    sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert sha == summary["receipt_sha256"], (sha, summary["receipt_sha256"])
    actual = {}
    tails = {}
    for step in (1, 3):
        per_layer = {}
        for layer in (0, 13, 26, 39):
            row = observed[f"{step}:{layer}"]
            raw, transported = row["fit_energy"]["raw"], row["fit_energy"]["transport"]
            per_layer[str(layer)] = {
                "raw_attention_squared_error": raw,
                "transported_attention_squared_error": transported,
                "attention_energy_ratio": transported / raw,
                "attention_l2_ratio": math.sqrt(transported / raw),
                "energy_change": transported - raw,
                "raw_fitting_vs_base": raw / row["base_energy"]["raw"],
                "transported_fitting_vs_base": transported / row["base_energy"]["transport"],
            }
        er = sum(x["raw_attention_squared_error"] for x in per_layer.values())
        et = sum(x["transported_attention_squared_error"] for x in per_layer.values())
        gm = geometric([x["attention_energy_ratio"] for x in per_layer.values()])
        assert abs(gm - summary["fit_replay"][str(step)]["fitted"]) < 1e-12
        actual[str(step)] = {
            "per_layer": per_layer,
            "geometric_mean_energy_ratio": gm,
            "geometric_mean_l2_ratio": math.sqrt(gm),
            "pooled_raw_energy": er,
            "pooled_transported_energy": et,
            "pooled_energy_ratio": et / er,
            "pooled_l2_ratio": math.sqrt(et / er),
            "raw_energy_layer_weights": {
                k: v["raw_attention_squared_error"] / er for k, v in per_layer.items()
            },
        }
        site = observed[f"{step}:39"]
        block = site["wan_wide8192_tail"]
        br, bt = block["held_block_energy"]["raw"], block["held_block_energy"]["transport"]
        ar, at = site["fit_energy"]["raw"], site["fit_energy"]["transport"]
        ratio = bt / br
        tails[str(step)] = {
            "layer": 39,
            "attention_energy_ratio": at / ar,
            "block_energy_ratio": ratio,
            "attention_l2_ratio": math.sqrt(at / ar),
            "block_l2_ratio": math.sqrt(ratio),
            "raw_finite_tail_squared_gain": br / ar,
            "transported_finite_tail_squared_gain": bt / at,
            "relative_finite_tail_squared_gain": (bt / at) / (br / ar),
            "block_energy_change": bt - br,
            "same_probe_ffn_inputs": all(block["probe_ffn_input_exact"].values()),
            "native_replay_vs_live_relative_l2": block["native_live_block_relative_l2"],
            "reference": "Native attention passed through the same fitted wide8192 FFN, not full-row Dense FFN",
        }
        if ratio < 1:
            # Let z_t=z_r+d. E_t<E_r implies <z_r,d><-||d||^2/2.
            # Among all d consistent with the two endpoint norms, the largest
            # possible cos(z_r,d) equals -sqrt(1-E_t/E_r).
            tails[str(step)]["finite_update_maximum_cosine_with_raw_tail_error"] = -math.sqrt(1-ratio)
            tails[str(step)]["finite_update_minimum_angle_degrees"] = math.degrees(math.acos(-math.sqrt(1-ratio)))

    rng = np.random.default_rng(20261126)
    n, m, rank, channels = 96, 64, 12, 8
    psi = rng.normal(size=(n, rank))
    p = np.arange(m)
    operator = psi @ np.linalg.solve(psi[p].T @ psi[p] + 6.4*np.eye(rank), psi[p].T)
    innovation = rng.normal(size=(n, channels))
    scale = np.exp(rng.normal(size=channels))
    normalized_then_decoded = (operator @ (innovation[p] / scale)) * scale
    direct = operator @ innovation[p]
    normalization_error = float(np.max(np.abs(normalized_then_decoded-direct)))
    assert normalization_error < 5e-14

    # Shared input/future readout resource. Each current readout has rank one;
    # independent positive diagonal scales give a rank-four family of readouts.
    w = np.ones(4)
    scales = np.ones((4, 4)) + np.eye(4)
    readouts = scales * w[None]
    u, singular, vh = np.linalg.svd(readouts, full_matrices=False)
    h = rng.normal(size=4)
    encode = vh
    decode = u @ np.diag(singular)
    reconstruction_error = float(np.max(np.abs(readouts @ h - decode @ (encode @ h))))
    assert np.linalg.matrix_rank(readouts) == 4 and reconstruction_error < 1e-13
    independent = np.zeros((2, 8)); independent[0, :4] = w; independent[1, 4:] = w
    resource = {
        "each_readout_rank": [int(np.linalg.matrix_rank(row[None])) for row in readouts],
        "two_known_future_scales_shared_state_minimum_rank": int(np.linalg.matrix_rank(readouts[:2])),
        "four_known_future_scales_shared_state_minimum_rank": int(np.linalg.matrix_rank(readouts)),
        "shared_readout_singular_values": singular.tolist(),
        "exact_shared_encoding_error": reconstruction_error,
        "same_readout_independent_input_direct_sum_rank": int(np.linalg.matrix_rank(independent)),
        "interpretation": "Rank-one current output does not imply one scalar suffices for an arbitrary future diagonal-scale family. Different rows/layers with independent states require a direct-sum treatment unless a variation constraint is justified.",
    }

    # Task-only recovery can preserve a readout without recovering full input.
    obs = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
    task = np.array([[1., 2., 0, 0], [-2., -4., 0, 0]])
    decoder = task @ np.linalg.pinv(obs)
    assert np.max(np.abs(task - decoder@obs)) < 1e-14
    invisible = np.array([0., 0, 2., -3.])
    bad_task = task.copy(); bad_task[0, 2] = 1.
    target_recovery = {
        "observation_rank": int(np.linalg.matrix_rank(obs)),
        "task_rank": int(np.linalg.matrix_rank(task)),
        "factorization_error": float(np.max(np.abs(task-decoder@obs))),
        "unobserved_full_state_variation_norm": float(np.linalg.norm(invisible)),
        "task_change_on_unobserved_variation": float(np.linalg.norm(task@invisible)),
        "changed_target_that_breaks_kernel_condition": float(np.linalg.norm(bad_task@invisible)),
    }

    # The two low39 endpoint-energy pairs can be produced by a purely linear
    # anisotropic map. Thus observed norm reversal establishes changed task
    # weighting, while nonlinearity or cancellation with the Dense-FFN error
    # is not identifiable from these four energy scalars.
    low = observed["3:39"]
    ar, at = low["fit_energy"]["raw"], low["fit_energy"]["transport"]
    br, bt = (low["wan_wide8192_tail"]["held_block_energy"][x] for x in ("raw", "transport"))
    e_raw = np.array([math.sqrt(ar), 0.])
    e_transport = np.array([0., math.sqrt(at)])
    tail_matrix = np.diag([math.sqrt(br/ar), math.sqrt(bt/at)])
    measured = [e_raw@e_raw, e_transport@e_transport,
                np.linalg.norm(tail_matrix@e_raw)**2,
                np.linalg.norm(tail_matrix@e_transport)**2]
    linear_witness = {
        "scope": "Synthetic 2D construction matching real endpoint energies; not a model of Wan weights",
        "input_errors": [e_raw.tolist(), e_transport.tolist()],
        "linear_tail": tail_matrix.tolist(),
        "maximum_matched_energy_absolute_error": float(np.max(np.abs(np.array(measured)-np.array([ar, at, br, bt])))),
    }
    assert linear_witness["maximum_matched_energy_absolute_error"] < 1e-10

    return {
        "evidence_commit": COMMIT,
        "receipt_sha256": sha,
        "actual_scope": "Recomputation of archived DEV0392, two experts x four audited layers, H448; one source, not eight independent samples",
        "new_model_or_gpu_runs": 0,
        "actual_energy_recomputation": actual,
        "actual_same_site_tail_comparison": tails,
        "synthetic_fixed_feature_normalization_identity_max_abs": normalization_error,
        "synthetic_shared_state_certificate": resource,
        "synthetic_task_observability_certificate": target_recovery,
        "synthetic_linear_anisotropy_witness": linear_witness,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    SOURCE = args.source
    result = run()
    output = HERE / "joint_readout_verified.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
