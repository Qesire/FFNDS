"""Recompute published scalar evidence and finite counterexamples; no ML dependencies.

Reads repository JSON/source bytes only. Does not load model tensors, access GPUs,
re-run a recorded experiment, or independently authenticate remote RUN payloads.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "research/20261028"
checks = []
bindings = []


def check(name, condition, details=None):
    if not condition:
        raise AssertionError(name)
    checks.append({"name": name, "passed": True, "details": details})


def close(a, b, rel=1e-10, abs_tol=1e-12):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def read(name):
    path = INPUT_DIR / (name + "_20261028.json")
    data = path.read_bytes()
    bindings.append({"path": path.relative_to(ROOT).as_posix(),
                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return json.loads(data)


def dot(x, y):
    return sum(a*b for a, b in zip(x, y))


def centered(x):
    mean = sum(x)/len(x)
    return [v-mean for v in x]


def sub(x, y):
    return [a-b for a, b in zip(x, y)]


def run():
    checks.clear()
    bindings.clear()
    fix = read("ORIGINAL_BF16_FIXED_NORM_TRANSFER_RESULTS")
    reductions = []
    for row in fix["same_state_comparison"]:
        reduction = 1-row["corrected_head_relative_SSE"]/row["bare_head_relative_SSE"]
        check("fixed_gain_step_" + str(row["step"]), close(reduction, row["reduction_fraction"]))
        reductions.append(reduction)
    own = fix["own_state_comparison"]
    own_reduction = 1-own["corrected_own_state_flow_relative_SSE"]/own["bare_own_state_flow_relative_SSE"]
    check("fixed_gain_own_state", close(own_reduction, own["reduction_fraction"]))
    check("source_scope_not_confirmation",
          fix["independent_native_transfer_image_clusters"] == 1
          and fix["source_role"] == "PREVIOUSLY_EXPOSED_DEVELOPMENT_NOT_NEW_HELDOUT"
          and fix["new_video_generations"] == 0)
    corners = [fix["four_corner"], fix["bare_four_corner"]]
    raw = read("ORIGINAL_BF16_LOW39_FOUR_CORNER_RESULTS")
    corners.extend(raw["four_corner"])
    for i, c in enumerate(corners):
        for part in [c] + c.get("time_slots", []):
            check("four_corner_closure_%d_%s" % (i, part.get("slot", "all")),
                  close(sum(part["terms"].values()), part["total_squared"], abs_tol=1e-9))
    corrected = fix["four_corner"]["relative_terms"]
    check("corrected_propagation_exceeds_local",
          corrected["Dense_state_propagation"] > corrected["teacher_state_local"])

    bh = read("BH_A0_MATCHED_TERMINAL_HEAD_RESULTS")
    bh_ratios = {}
    for expert, parts in bh["source_summary"].items():
        h = parts["heldout8_exposed"]
        rows = list(h["per_source"].values())
        means = {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}
        check("BH_source_aggregation_" + expert,
              all(close(v, h["equal_source_mean_head_relative_SSE"][k]) for k, v in means.items()))
        check("BH_zero_wins_" + expert,
              all(r["BH_saved"] > r["energy94"] and r["BH_saved"] > r["random94"] for r in rows)
              and len(rows) == 8)
        bh_ratios[expert] = means["BH_saved"]/means["energy94"]

    bf = read("BF_FROZEN6016_TERMINAL_RESULTS")
    bf_metrics = []
    for expert, bases in bf["results"].items():
        for basis, parts in bases.items():
            h = parts["heldout8_exposed"]
            f = h["final"]["equal_source_mean"]
            initial = h["initial"]["equal_source_mean"]["head_relative_SSE"]
            oracle = h["oracle"]["equal_source_mean"]["head_relative_SSE"]
            a0 = h["A0_energy94"]["head_relative_SSE"]
            for stage in ("initial", "final", "oracle"):
                s = h[stage]["equal_source_mean"]
                check("BF_signed_closure_" + expert + "_" + basis + "_" + stage,
                      close(s["direct_relative_SSE"]+s["radial_relative_SSE"]
                            +s["signed_cross_relative_SSE"], s["continuous_relative_SSE"])
                      and close(s["continuous_relative_SSE"]+s["BF16_remainder_relative_SSE"]
                                +s["BF16_signed_remainder_cross"], s["head_relative_SSE"]))
            check("BF_numerator_larger_" + expert + "_" + basis,
                  f["direct_relative_SSE"] > f["radial_relative_SSE"])
            bf_metrics.append({"expert": expert, "basis": basis,
                               "head": f["head_relative_SSE"], "A0": a0, "initial": initial,
                               "relative_reduction_vs_A0": 1-f["head_relative_SSE"]/a0,
                               "final_over_teacher_projection": f["head_relative_SSE"]/oracle,
                               "direct_to_radial_squared_ratio": f["direct_relative_SSE"]/f["radial_relative_SSE"]})
    cost = bf["static_cost"]
    mac_ratio = cost["BF_MAC_per_row"]/cost["A0_MAC_per_row"]
    check("BF_logical_MAC_ratio", close(mac_ratio, cost["BF_to_A0_MAC_ratio"]))

    quality = read("COMPACT6016_CONSUMER_QUALITY_RESULTS")
    raft = []
    for source, modes in quality["continuous_RAFT"].items():
        for mode, values in modes.items():
            ratio = values["candidate_mean"]/values["dense_mean"]
            check("RAFT_ratio_" + source + "_" + mode, close(ratio, values["ratio"]))
            raft.append({"source": source, "mode": mode, "ratio": ratio})
    check("quality_not_accepted", quality["quality_accepted"] is False
          and quality["human_fullplay_review"] == "NOT_RUN")
    steps = read("COMPACT6016_STEP_WEIGHT_SCALAR_AUDIT")
    schedule_ratio = sum(steps["delta_sigma"])/steps["delta_sigma"][0]
    check("constant_aligned_bias_schedule_ratio", close(schedule_ratio, 16.0))
    trajectory = []
    for g in steps["groups"]:
        n = g["cumulative_error_norm_divided_by_cumulative_step"]
        trajectory.append({"source": g["source"], "mode": g["mode"],
                           "step_normalized_growth": n[-1]/n[0],
                           "raw_error_norm_growth": schedule_ratio*n[-1]/n[0]})

    # Exact continuous variance identity over all masks of a signed finite bundle.
    u = [2., -1., 3., .5]
    cols = [[1., -.4, .7, 0.], [-.8, .2, -.9, .3], [0., 1., .2, -.4]]
    z = centered(u)
    residuals = []
    for mask in itertools.product((0, 1), repeat=3):
        removed = [sum(mask[j]*cols[j][i] for j in range(3)) for i in range(4)]
        cr = centered(removed)
        zs = centered(sub(u, removed))
        actual = (dot(zs, zs)-dot(z, z))/4
        predicted = (-2*dot(z, cr)+dot(cr, cr))/4
        residuals.append(abs(actual-predicted))
    check("all_8_masks_signed_variance_identity", max(residuals) < 1e-12)

    # A linear-head null direction can change its finite normalized output.
    s = 1/math.sqrt(2)
    e1, e2 = [s, -s, 0., 0.], [0., 0., s, -s]
    dense = [a+b for a,b in zip(e1, e2)]
    sparse = e1
    eps = 1e-6
    f = lambda v: dot(e1, v)/math.sqrt(dot(v, v)/4+eps)
    check("linear_null_not_normalized_null",
          close(dot(e1, sub(dense, sparse)), 0.) and abs(f(dense)-f(sparse)) > .5,
          {"linear_error": dot(e1, sub(dense, sparse)),
           "finite_normalized_error": f(sparse)-f(dense)})

    # Weighted scalar optimum is not generally reciprocal(mean sigma ratio).
    a_s, a_d, w = [2., 4.], [1., 3.], [.5, .5]
    alpha = sum(t*x*y for t,x,y in zip(w,a_s,a_d))/sum(t*x*x for t,x in zip(w,a_s))
    loss = lambda a: sum(t*(a*x-y)**2 for t,x,y in zip(w,a_s,a_d))
    naive = 1/(sum(x/y for x,y in zip(a_s,a_d))/2)
    check("fixed_gain_weighted_optimum", loss(alpha) < loss(naive)
          and loss(alpha) <= loss(alpha-.01) and loss(alpha) <= loss(alpha+.01),
          {"optimal_alpha": alpha, "reciprocal_mean_ratio": naive})

    # Reference E4M3/E2M1 positive nearest rounding; deterministic values avoid ties.
    # This is a format-level counterexample, not a native kernel equivalence test.
    fp8 = sorted(set([m*2**-9 for m in range(8)] +
                    [2**e*(1+m/8) for e in range(-6,9) for m in range(8)
                     if 2**e*(1+m/8) <= 448]))
    fp4 = [0., .5, 1., 1.5, 2., 3., 4., 6.]
    nearest = lambda x, levels: min(levels, key=lambda y: abs(y-x))
    def quant_block(block, global_scale):
        local = nearest(max(abs(v) for v in block)/(6*global_scale), fp8)
        scale = local*global_scale
        return [0. if scale == 0 else math.copysign(nearest(abs(v)/scale, fp4)*scale, v)
                for v in block]
    block = [1., .27, -.36, .73]*4
    other = [100.3]*16
    gd = max(map(abs, block+other))/(448*6)
    gs = max(map(abs, block))/(448*6)
    full = quant_block(block, gd)+quant_block(other, gd)
    kept_frozen = quant_block(block, gd)
    kept_recomputed = quant_block(block, gs)
    check("NVFP4_frozen_global_local_closure", kept_frozen == full[:16])
    check("NVFP4_recomputed_global_counterexample", kept_recomputed != full[:16],
          {"dense_global_scale": gd, "selected_global_scale": gs,
           "max_retained_difference": max(abs(a-b) for a,b in zip(kept_recomputed,full[:16]))})

    return {"schema": "ffnds.research_route_review_scalar_audit.v1", "date": "2026-10-29",
            "review_basis_commit": "a11ddcf5658a4179da42681fb31da11426b1c4cc",
            "evidence": "STATIC_PUBLISHED_SCALAR_RECOMPUTATION_AND_FINITE_ALG",
            "new_model_calls": 0, "new_GPU_calls": 0, "model_tensor_payload_reads": 0,
            "remote_RUN_payloads_reauthenticated": False, "input_bindings": bindings,
            "fixed_gain_same_state_reductions": reductions, "fixed_gain_own_state_reduction": own_reduction,
            "BH_over_A0_energy": bh_ratios, "BF_metrics": bf_metrics, "BF_logical_MAC_ratio": mac_ratio,
            "quality_MUSIQ": quality["equal_source_common_six"]["imaging_quality"],
            "RAFT_ratios": raft, "constant_bias_schedule_growth": schedule_ratio,
            "trajectory_growth": trajectory, "check_count": len(checks), "checks": checks,
            "limitations": "No fresh statistical confirmation, native kernel validation, video playback or speed measurement."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("evidence", "check_count", "BF_logical_MAC_ratio",
                       "fixed_gain_same_state_reductions", "fixed_gain_own_state_reduction")},
                     ensure_ascii=False))

