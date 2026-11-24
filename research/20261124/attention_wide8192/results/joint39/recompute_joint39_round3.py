#!/usr/bin/env python3
"""Recompute published joint39 receipts; this does not rerun a model/vector fit.

Default inputs are the pinned source receipts next to this script.
All aggregation is within background/expert over two source identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HEAD = "0a7150ca9e260a07aa7b0fa1ccded9c2be3e342c"
VARIANTS = ("phi48", "head112", "ffn112", "D176")
METRICS = ("point", "spatial", "temporal")
KINDS = ("P64_fit", "H448_OLS_floor", "H448_same_ridge")


def gm(values):
    values = list(values)
    return math.exp(sum(map(math.log, values)) / len(values))


def load(path):
    return json.loads(path.read_text())


def run(source_dir, old_source_dir):
    replay = load(source_dir / "D176_REPLAY.json")
    summary = load(source_dir / "D176_SUMMARY.json")
    sites = replay["sites"]
    assert len(sites) == 8
    assert {s["source"] for s in sites} == {"vbi_0392", "vbi_1013"}
    report = dict(
        pinned_head=HEAD,
        execution_scope="Independent CPU arithmetic over published JSON receipts; no tensors, model, or GPU execution.",
        statistical_units="Two DEV sources; background and expert are nested repeated conditions, not eight independent sources.",
        aggregates={}, source_sites=[], module_response=[],
        diagnostic_limits={
            "H448_floor": "OLS fitted and evaluated on H448, not a P64-feasible predictor or a population lower bound.",
            "P64_gap": "P64 energy minus H448 OLS energy is total excess over the full H448 dictionary. It combines the P-induced reachable subspace, regularization, and fitting effects; it is not identified sampling variance.",
            "H448_ridge_gap": "H448 ridge minus H448 OLS isolates the effect of this regularized H448 fit, but subtracting it from P64 excess is not an orthogonal stochastic bias/variance decomposition.",
            "rounding": "The stored P64 prediction is rounded to FP32; H448 fits remain FP64. Exact projection identities apply before prediction rounding.",
            "edge_floor": "H448 OLS minimizes point energy; the associated spatial/temporal energies are not edge-optimal lower bounds.",
        },
    )
    verification_errors = []
    for background in ("Dense", "wide8192"):
        report["aggregates"][background] = {}
        for expert in ("high", "low"):
            subset = [s for s in sites if s["background"] == background and s["expert"] == expert]
            assert len(subset) == 2
            group = {}
            for base in ("cheap", "B11_fixed"):
                agg = {"variants": {}, "contrasts": {}}
                for variant in VARIANTS:
                    means = {
                        kind: {
                            m: gm(s["scores"][base][variant][kind][m] / s["scores"]["B11_fixed"]["base"][m] for s in subset)
                            for m in METRICS
                        }
                        for kind in KINDS
                    }
                    means["P64_over_full_H448_point_floor"] = gm(
                        s["scores"][base][variant]["P64_fit"]["point"] /
                        s["scores"][base][variant]["H448_OLS_floor"]["point"] for s in subset
                    )
                    means["source_P64_point_ratios"] = {
                        s["source"]: s["scores"][base][variant]["P64_fit"]["point"] /
                        s["scores"]["B11_fixed"]["base"]["point"] for s in subset
                    }
                    agg["variants"][variant] = means
                    expected = summary["by_background"][background][expert][base][variant]
                    verification_errors.extend(abs(means["P64_fit"][m] - expected["P64_ratio"][m]) for m in METRICS)
                    verification_errors.append(abs(means["H448_OLS_floor"]["point"] - expected["H448_OLS_point_ratio"]))
                for larger, smaller in (("head112", "phi48"), ("ffn112", "phi48"), ("D176", "phi48"), ("D176", "ffn112"), ("D176", "head112")):
                    agg["contrasts"][f"{larger}_over_{smaller}"] = {
                        kind: {
                            m: gm(s["scores"][base][larger][kind][m] / s["scores"][base][smaller][kind][m] for s in subset)
                            for m in METRICS
                        }
                        for kind in KINDS
                    }
                group[base] = agg
            group["baseline_point_ratios"] = {
                "independent_over_cheap": gm(s["scores"]["B11_fixed"]["base"]["point"] / s["scores"]["cheap"]["base"]["point"] for s in subset),
                "D176_cheap_floor_over_independent_base_floor": gm(s["scores"]["cheap"]["D176"]["H448_OLS_floor"]["point"] / s["scores"]["B11_fixed"]["D176"]["H448_OLS_floor"]["point"] for s in subset),
            }
            report["aggregates"][background][expert] = group
    for s in sites:
        r = {k: s[k] for k in ("source", "background", "expert")}
        r["baselines"] = {}
        for base in ("cheap", "B11_fixed"):
            d = s["scores"][base]
            norm = s["scores"]["B11_fixed"]["base"]["point"]
            variants = {}
            for variant in VARIANTS:
                p, o, h = (d[variant][k]["point"] for k in KINDS)
                assert p >= o and h >= o
                variants[variant] = {
                    "point_energies": {"P64": p, "H448_OLS": o, "H448_ridge": h},
                    "point_ratios_to_independent": {"P64": p/norm, "H448_OLS": o/norm, "H448_ridge": h/norm},
                    "full_dictionary_excess_energy": p-o,
                    "full_dictionary_excess_share_of_P64_energy": (p-o)/p,
                    "H448_regularized_excess_energy": h-o,
                    "P64_minus_H448_regularized_energy": p-h,
                    "P64_rank": d[variant]["P64_rank"],
                }
            # These differences are paired descriptive contrasts, not causal
            # allocations of independent contribution.
            interaction = {
                kind: {m: d["D176"][kind][m] - d["head112"][kind][m] - d["ffn112"][kind][m] + d["phi48"][kind][m] for m in METRICS}
                for kind in KINDS
            }
            verification_errors.extend(abs(interaction["P64_fit"][m] - d["P64_addition_interaction"][m]) for m in METRICS)
            r["baselines"][base] = {"variants": variants, "addition_interaction": interaction}
        # Same D176 dictionary is used for both baselines. For Q = I - Pi_DH,
        # r_c = truth - cheap, r_i = truth - independent, and
        # delta_b = independent - cheap, Q delta_b = Q r_c - Q r_i.
        # Reverse/ordinary triangle inequalities give rigorous bounds without
        # the unavailable cross inner product. These are per-site bounds.
        ec = s["scores"]["cheap"]["D176"]["H448_OLS_floor"]["point"]
        ei = s["scores"]["B11_fixed"]["D176"]["H448_OLS_floor"]["point"]
        eind = s["scores"]["B11_fixed"]["base"]["point"]
        lower, upper = (math.sqrt(ec)-math.sqrt(ei))**2, (math.sqrt(ec)+math.sqrt(ei))**2
        r["D176_baseline_difference_outside_dictionary_bounds"] = {
            "meaning": "Energy of (I - Pi_DH)(B11_fixed - cheap), with same D176 and H448 point Euclidean projector.",
            "cheap_floor_energy": ec,
            "independent_base_floor_energy": ei,
            "lower_energy": lower,
            "upper_energy": upper,
            "lower_over_independent_point_energy": lower/eind,
            "upper_over_independent_point_energy": upper/eind,
            "validity": "Reverse triangle bounds from published FP64 least-squares residual norms; not an information-theoretic bound for nonlinear functions of the raw features.",
        }
        report["source_sites"].append(r)
    gram_errors = []
    for name, path in (("vbi_0392", old_source_dir/"RESULT_0392.json"), ("vbi_1013", source_dir/"RESULT_1013.json")):
        original = load(path)
        for background, bg in original["results"].items():
            for expert, site in bg["experts"].items():
                bgra, fgra = (site["branch_gram"][k] for k in ("block_exit", "flow"))
                gb, gf = bgra["gram"], fgra["gram"]
                for g, info in ((gb,bgra),(gf,fgra)):
                    gram_errors.append(abs(sum(map(sum,g))-info["B11_error"]))
                r = dict(source=name, background=background, expert=expert)
                r["attention_before_ffn_energy"] = site["prefix_response"]["T"]
                r["attention_after_native_ffn_energy"] = gb[0][0]
                r["native_ffn_attenuation_fraction"] = 1-gb[0][0]/site["prefix_response"]["T"]
                r["head_F_over_A_direction_energy_gain_ratio"] = (gf[1][1]/gb[1][1])/(gf[0][0]/gb[0][0])
                r["flow_Gram"] = gf
                r["flow_EF_over_EA"] = gf[1][1]/gf[0][0]
                r["flow_I_over_EA"] = gf[2][2]/gf[0][0]
                r["flow_AF_cosine"] = gf[0][1]/math.sqrt(gf[0][0]*gf[1][1])
                r["flow_sum_cross_over_total"] = (sum(map(sum,gf))-sum(gf[i][i] for i in range(3)))/fgra["B11_error"]
                r["flow_joint_over_sum_single"] = fgra["B11_error"]/(gf[0][0]+gf[1][1])
                r["FFN_refit_energy_change_fraction"] = site["flow_scores"]["B11_refit"]["point"]/site["flow_scores"]["B11_fixed"]["point"]-1
                report["module_response"].append(r)
    report["verification"] = {
        "max_summary_or_interaction_absolute_error": max(verification_errors),
        "max_Gram_closure_absolute_error": max(gram_errors),
        "all_eight_native_FFN_attenuate_measured_attention_perturbation": all(r["native_ffn_attenuation_fraction"]>0 for r in report["module_response"]),
        "all_eight_flow_attention_FFN_direct_inner_products_positive": all(r["flow_Gram"][0][1]>0 for r in report["module_response"]),
        "all_eight_flow_I_inner_products_with_A_and_F_negative": all(r["flow_Gram"][0][2]<0 and r["flow_Gram"][1][2]<0 for r in report["module_response"]),
    }
    assert report["verification"]["max_summary_or_interaction_absolute_error"] < 1e-10
    assert report["verification"]["max_Gram_closure_absolute_error"] < 1e-7
    files = [source_dir/n for n in ("D176_REPLAY.json", "D176_SUMMARY.json", "RESULT_1013.json")] + [old_source_dir/"RESULT_0392.json"]
    report["source_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=here)
    parser.add_argument("--old-source-dir", type=Path, default=here)
    parser.add_argument("--output", type=Path, default=here/"joint39_round3_recomputed.json")
    args = parser.parse_args()
    result = run(args.source_dir,args.old_source_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result["verification"],ensure_ascii=False,indent=2))
