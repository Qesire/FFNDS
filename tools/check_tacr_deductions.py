"""Reproducible TACR algebra examples and saved R16/R17 scalar readout.

Python standard library only. Executes no model, tensor workload or training.
Toy counterexamples establish logical possibilities; repository readouts retain
their original one-source scope. Run from repository root:
  python tools/check_tacr_deductions.py
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def loss(e, v):
    r = sub(e, v)
    return dot(r, r)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main(root):
    checks = []

    # A compensation increment can reverse the sign of the executed action.
    e, d, dc = [1.0], [1.0], [2.0]
    net = sub(d, dc)
    close(loss(e, d), 0)
    close(loss(e, net), 4)
    checks.append(dict(id="net_action_sign", raw_predicted_loss=0,
                       actual_net_loss=4, baseline_loss=1))

    # The proposal's example is a nonmonotone quadratic subset problem.
    e, v = [1., 0.], [[3., 1.], [-2., -1.]]
    rows = []
    for b in itertools.product([0, 1], repeat=2):
        vb = [sum(b[j] * v[j][k] for j in range(2)) for k in range(2)]
        direct = loss(e, vb)
        quadratic = dot(e, e) - 2 * sum(b[j] * dot(v[j], e) for j in range(2))
        quadratic += sum(b[j] * b[k] * dot(v[j], v[k]) for j in range(2) for k in range(2))
        close(direct, quadratic)
        rows.append(dict(mask=list(b), loss=direct))
    close(rows[-1]["loss"], 0)
    assert rows[1]["loss"] > 1 and rows[2]["loss"] > 1
    checks.append(dict(id="signed_pair_quadratic", cases=rows,
                       pair_gain_interaction=-2 * dot(*v)))

    # C(empty)=0 alone imposes neither additivity nor dense-endpoint closure.
    c = {(0, 0): 0., (1, 0): 0., (0, 1): 0., (1, 1): 3.}
    delta_pair = 2 - c[1, 1]
    sum_single = (1 - c[1, 0]) + (1 - c[0, 1])
    close(delta_pair, -1)
    close(sum_single, 2)
    checks.append(dict(id="mask_compensation_nonadditive", pair_delta=delta_pair,
                       sum_single_deltas=sum_single))

    # The e-v covariance is decisive even when both conditional means vanish.
    samples = [(-1., -1.), (1., 1.)]
    baseline = sum(e * e for e, _ in samples) / 2
    after = sum((e - v) ** 2 for e, v in samples) / 2
    mean_e = sum(e for e, _ in samples) / 2
    mean_v = sum(v for _, v in samples) / 2
    plug_in_gain = 2 * mean_e * mean_v - mean_v ** 2
    true_gain = sum(2 * e * v - v * v for e, v in samples) / 2
    close(true_gain, 1)
    close(plug_in_gain, 0)
    checks.append(dict(id="conditional_covariance", baseline=baseline, after=after,
                       true_gain=true_gain, mean_vector_plugin_gain=plug_in_gain))

    # No signed means are needed to make state-conditioned routing useful.
    rows = []
    for z, sign in itertools.product([0, 1], [-1., 1.]):
        e = sign
        values = [sign if z == 0 else 0., sign if z == 1 else 0.]
        rows.append(dict(z=z, e=e, values=values))
    static_losses = [sum((r["e"] - r["values"][a]) ** 2 for r in rows) / 4 for a in [0, 1]]
    dynamic_loss = sum((r["e"] - r["values"][r["z"]]) ** 2 for r in rows) / 4
    close(min(static_losses), .5)
    close(dynamic_loss, 0)
    checks.append(dict(id="zero_mean_conditional_risk_router", best_fixed_loss=min(static_losses),
                       conditional_loss=dynamic_loss, conditional_action_means=[0., 0.]))

    # Projection onto an energetic but irrelevant axis loses the ordering.
    e, good, bad = [0., 1.], [0., 1.], [0., -1.]
    close(loss(e, good), 0)
    close(loss(e, bad), 4)
    close(loss(e[:1], good[:1]), loss(e[:1], bad[:1]))
    checks.append(dict(id="projection_nullspace", projected_losses=[0., 0.],
                       full_losses=[0., 4.]))

    # Finite nonlinear responses need not agree with a Jacobian at the anchor.
    f = lambda x: x * x
    single_a, single_b, pair = f(1)-f(0), f(-1)-f(0), f(0)-f(0)
    interaction = pair - single_a - single_b
    close(interaction, -2)
    checks.append(dict(id="finite_response_interaction", jacobian_at_anchor=0,
                       single_responses=[single_a, single_b], pair_response=pair,
                       interaction=interaction))

    # Eliminate a free compensator span U=(1,0), Q=I: K=diag(0,1).
    e, v = [1., 2.], [2., 1.]
    profiled = []
    for b in [0, 1]:
        residual = sub(e, [b * x for x in v])
        a_star = residual[0]
        direct = loss(residual, [a_star, 0.])
        schur = residual[1] ** 2
        close(direct, schur)
        profiled.append(direct)
    checks.append(dict(id="compensation_quotient", profiled_losses=profiled,
                       eliminated_axis="span((1,0)); toy offline optimum"))

    # A finite action-value predictor's error bounds selection regret.
    value, predicted, eps = [1., .9, -.2], [.94, .99, -.15], .1
    assert max(abs(a-b) for a, b in zip(value, predicted)) <= eps
    selected = max(range(3), key=lambda j: predicted[j])
    regret = max(value) - value[selected]
    assert regret <= 2 * eps + 1e-12
    checks.append(dict(id="action_value_regret", uniform_error_bound=eps,
                       selected=selected, regret=regret, regret_bound=2*eps))

    groups = [54, 108, 108, 162]
    base = sum(groups) / (4 * 216)
    added = 27 / 216
    rr = 256 / 13824
    close(base, .5)
    close(base + added, .625)
    checks.append(dict(id="projection_MAC_accounting", actual_base_groups_by_step=groups,
                       weighted_base_fraction=base, illustrative_extra_groups_per_site=27,
                       illustrative_augmented_main_fraction=base+added,
                       R256_two_projection_fraction=rr,
                       augmented_plus_R256_fraction=base+added+rr,
                       BF16_per_token_all216_r16_prediction_MiB=31668*216*16*2/2**20,
                       h256_to_all216_r16_linear_MAC_fraction=256*216*16/(2*5120*13824)))

    artifacts = ["research/20261023/R16V0_analysis.json",
                 "research/20261023/R17E0_system_timing_prefix.json"]
    data = [json.loads((root / p).read_text(encoding="utf-8")) for p in artifacts]
    r16, r17 = data
    arms = list(r17["per_arm"])
    timings = {}
    for arm in arms:
        stat = r17["per_arm"][arm]["complete_generation_wall_seconds"]
        timings[arm] = median(stat["all_five_values"])
        close(timings[arm], stat["median"])
    sse = {a: sum(p["same_input_native_SSE"][a] for p in r16["per_position"]) for a in arms}
    fixed = min(sse[a] for a in ("fixed_stride", "fixed_hash"))
    gain = 1 - sse["dynamic_residual"] / fixed
    close(gain, r16["dynamic_native_local_gain_over_best_single_fixed"])
    rgb = {a: r16["RGB_fidelity_to_dense"][a]["pooled_RGB_MSE_0_255"] for a in arms}
    rgb_ratio = rgb["dynamic_residual"] / min(rgb[a] for a in ("fixed_stride", "fixed_hash"))
    close(rgb_ratio, r16["dynamic_pooled_RGB_MSE_over_best_pixel_fixed"])
    ffn_s = r17["per_arm"]["dense_DQ"]["full160_FFN_including_all_shared_modulation_cuda_ms"]["median"] / 1000
    frac = ffn_s / timings["dense_DQ"]
    readout = dict(id="saved_R16_R17_recalculation", scope="one old source; five R17 technical repeats per arm",
                   independent_sources=r17["independent_performance_sources"],
                   R16_local_gain=gain, R16_RGB_MSE_ratio=rgb_ratio,
                   R17_median_seconds=timings,
                   dense_over_dynamic=timings["dense_DQ"]/timings["dynamic_residual"],
                   dense_over_UB1=timings["dense_DQ"]/timings["UB1"],
                   dynamic_over_UB1=timings["dynamic_residual"]/timings["UB1"],
                   dense_FFN_CUDA_ms_to_wall_fraction=frac,
                   conditional_Amdahl_infinite_FFN_speedup=1/(1-frac),
                   Amdahl_assumption="Illustrative serial additive decomposition; CUDA/wall measurement is not an identified causal split.")
    checks.append(readout)
    return dict(id="TACR_CPU_DEDUCTIONS_20261023", status="COMPLETED", schema_version=1,
                source_commit="27f86cd3f8821b2771f1be633baf32d4cc79ebb0",
                provenance="Ten synthetic algebra checks plus one saved-summary scalar recomputation; zero new model calls.",
                check_count=len(checks), new_model_calls=0, training_steps=0,
                input_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in artifacts},
                checks=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = main(args.root)
    output = args.output or args.root / "research/20261023/TACR_cpu_deductions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(dict(status=result["status"], check_count=result["check_count"],
                          readout=result["checks"][-1]), ensure_ascii=False, indent=2))
