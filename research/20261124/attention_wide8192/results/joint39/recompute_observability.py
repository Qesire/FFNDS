#!/usr/bin/env python3
"""Arithmetic on the published D176 observability JSON; no tensor/model rerun."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import math
from pathlib import Path

import numpy as np

HEAD = "cc01675282db463fc8983f9113ea3a4ff8c309ef"
VARIANTS = ("phi48", "head112", "ffn112", "D176")
BASES = ("cheap", "B11_fixed")


def gm(values):
    vals = list(values)
    return math.exp(sum(map(math.log, vals)) / len(vals))


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def run(source, prior):
    raw = json.loads(payload(source))
    old = json.loads(payload(prior))
    assert len(raw["sites"]) == 8
    old_sites = {(s["source"],s["background"],s["expert"]):s for s in old["sites"]}
    rows, geometry = [], []
    errors = dict(original_score=0.0, point_split=0.0, spectrum_df=0.0,
                  gram_condition_relative=0.0, floor_split=0.0, rounding_norm=0.0,
                  producer_check_relative=0.0)
    for site in raw["sites"]:
        ident = {k:site[k] for k in ("source","background","expert")}
        old_site = old_sites[tuple(ident[k] for k in ("source","background","expert"))]
        independent_energy = site["scores"]["B11_fixed"]["base"]["point"]
        for base in BASES:
            for variant in VARIANTS:
                values = site["scores"][base][variant]
                before = old_site["scores"][base][variant]
                for category in ("P64_fit","H448_OLS_floor","H448_same_ridge"):
                    for metric in ("point","spatial","temporal"):
                        errors["original_score"] = max(errors["original_score"],
                            abs(values[category][metric]-before[category][metric]))
                info = values["information_diagnostic"]
                es = info["point_error"]
                budget = info["orthogonal_error_budget_FP64"]
                b, full, reachable, actual = (es[k] for k in
                    ("base","full_H_OLS","P_span_H_OLS","actual_P_ridge_FP64"))
                errors["point_split"] = max(errors["point_split"],
                    abs(actual-budget["outside_full_dictionary"]-
                        budget["full_dictionary_to_P_representer_span"]-
                        budget["P_span_coefficient_gap"]))
                errors["floor_split"] = max(errors["floor_split"],
                    abs(reachable-full-budget["full_dictionary_to_P_representer_span"]))
                errors["rounding_norm"] = max(errors["rounding_norm"],
                    abs(es["actual_P_ridge_FP32_prediction"]-actual)/independent_energy)
                errors["producer_check_relative"] = max(errors["producer_check_relative"],
                    info["producer_point_energy_check_relative"])
                useful, coefficient_gap = b-reachable, actual-reachable
                assert useful >= -1e-9 and coefficient_gap >= -1e-9
                # Let v = Pi_V r and c be actual correction. The report gives
                # ||v||^2 and ||v-c||^2, but not ||c||^2 or <r,c>.
                correction_lo = (math.sqrt(max(0,useful))-math.sqrt(max(0,coefficient_gap)))**2
                correction_hi = (math.sqrt(max(0,useful))+math.sqrt(max(0,coefficient_gap)))**2
                inner_lo = (useful+correction_lo-coefficient_gap)/2
                inner_hi = (useful+correction_hi-coefficient_gap)/2
                row = dict(**ident,base=base,variant=variant,
                    independent_point_energy=independent_energy,
                    energies=es,
                    ratios_to_independent={k:v/independent_energy for k,v in es.items()},
                    additive_budget_over_independent={
                        "full_dictionary":full/independent_energy,
                        "dictionary_to_Pspan":(reachable-full)/independent_energy,
                        "Pspan_coefficient":coefficient_gap/independent_energy},
                    recoverable_target_energy=useful,
                    recoverable_target_fraction=useful/b,
                    coefficient_gap_over_recoverable_target=coefficient_gap/useful if useful>0 else None,
                    actual_minus_zero_correction=actual-b,
                    correction_geometry_bounds={
                        "correction_energy_lower":correction_lo,
                        "correction_energy_upper":correction_hi,
                        "target_correction_inner_lower":inner_lo,
                        "target_correction_inner_upper":inner_hi,
                        "inner_sign_identified":inner_lo>0 or inner_hi<0,
                        "interpretation":"Bounds derive only from published norms. Actual correction energy/inner product and an H line-search optimum were not saved."},
                    ranks=info["ranks"])
                rows.append(row)
                if base == "B11_fixed":
                    s = np.asarray(info["singular_values"]["D_P"])
                    sh = np.asarray(info["singular_values"]["D_H"])
                    sr = np.asarray(info["singular_values"]["P_restriction_of_orthonormal_function_basis"])
                    reg = info["stability"]["n_lambda"]
                    filters = s*s/(s*s+reg)
                    edf = float(filters.sum())
                    st = info["stability"]
                    condition = (s[0]**2+reg)/(s[-1]**2+reg) if len(s)==64 else (s[0]**2+reg)/reg
                    errors["spectrum_df"]=max(errors["spectrum_df"],abs(edf-st["effective_probe_degrees_of_freedom"]))
                    errors["gram_condition_relative"]=max(errors["gram_condition_relative"],
                        abs(condition-st["regularized_probe_gram_condition"])/st["regularized_probe_gram_condition"])
                    record=dict(**ident,variant=variant,
                        ranks=info["ranks"],
                        P_singular_min=float(s[-1]),P_singular_max=float(s[0]),
                        ridge_filter_min=float(filters.min()),ridge_filter_max=float(filters.max()),
                        ridge_modes_ge_half=int((filters>=0.5).sum()),
                        ridge_modes_ge_nine_tenths=int((filters>=0.9).sum()),
                        ridge_modes_le_tenth=int((filters<=0.1).sum()),
                        effective_probe_df=edf,
                        regularized_probe_gram_condition=condition,
                        P_to_H_total_norm=st["P_to_H_ridge_operator_norm"],
                        P_to_H_per_query_RMS_norm=st["P_to_H_ridge_operator_norm"]/math.sqrt(7),
                        P_restriction_min_nonzero=float(sr[-1]),
                        P_restriction_max=float(sr[0]),
                        canonical_minimum_function_norm_P_to_H_total_gain=float(math.sqrt(1/sr[-1]**2-1)),
                        H_design_fractional_energy_top64=float((sh[:64]**2).sum()/(sh**2).sum()),
                        H_design_min_to_max=float(sh[-1]/sh[0]),
                        limitations="Singular values do not contain target alignment with singular vectors; spectra alone cannot allocate the coefficient gap to shrinkage or label mismatch.")
                    geometry.append(record)
                    other = site["scores"]["cheap"][variant]["information_diagnostic"]
                    assert other["singular_values"]==info["singular_values"]
                    assert other["ranks"]==info["ranks"]
    aggregates = {}
    for background in ("Dense","wide8192"):
        aggregates[background]={}
        for expert in ("high","low"):
            group={}
            for base in BASES:
                group[base]={}
                for variant in VARIANTS:
                    subset=[r for r in rows if r["background"]==background and r["expert"]==expert and r["base"]==base and r["variant"]==variant]
                    group[base][variant]=dict(
                        geometric_ratios={k:gm(r["ratios_to_independent"][k] for r in subset) for k in subset[0]["ratios_to_independent"]},
                        arithmetic_additive_budget={k:sum(r["additive_budget_over_independent"][k] for r in subset)/2 for k in subset[0]["additive_budget_over_independent"]},
                        recoverable_fraction_range=[min(r["recoverable_target_fraction"] for r in subset),max(r["recoverable_target_fraction"] for r in subset)],
                        coefficient_gap_over_useful_range=[min(r["coefficient_gap_over_recoverable_target"] for r in subset),max(r["coefficient_gap_over_recoverable_target"] for r in subset)])
            contrasts={}
            for larger, smaller in (("D176","ffn112"),("ffn112","phi48"),("head112","phi48"),("D176","phi48")):
                contrasts[larger+"_over_"+smaller]={}
                for base in BASES:
                    a=group[base][larger]["geometric_ratios"]
                    b=group[base][smaller]["geometric_ratios"]
                    contrasts[larger+"_over_"+smaller][base]={k:a[k]/b[k] for k in
                        ("full_H_OLS","P_span_H_OLS","actual_P_ridge_FP64")}
            group["contrasts"]=contrasts
            aggregates[background][expert]=group
    # A descriptive single-step consequence, not a full-trajectory error.
    one_step=[]
    weights={"high":(5/48)**2,"low":(5/8)**2}
    for r in rows:
        if r["background"]=="wide8192" and r["base"]=="B11_fixed" and r["variant"]=="D176":
            w=weights[r["expert"]]
            one_step.append({k:r[k] for k in ("source","expert")}|{
                "h_squared":w,
                "base_direct_latent_error":w*r["energies"]["base"],
                "Pspan_oracle_direct_latent_error":w*r["energies"]["P_span_H_OLS"],
                "available_Pspan_oracle_direct_reduction":w*r["recoverable_target_energy"],
                "actual_direct_latent_error":w*r["energies"]["actual_P_ridge_FP64"],
                "scope":"Frozen-input direct contribution at the recorded hit; ignores later feedback and cross-step error correlations."})
    assert errors["original_score"]<1e-10
    assert errors["point_split"]<1e-9 and errors["floor_split"]<1e-9
    assert errors["spectrum_df"]<1e-9 and errors["gram_condition_relative"]<1e-9
    return dict(
        pinned_head=HEAD,
        execution_scope="Independent arithmetic and spectrum postprocessing of real published receipts; no model execution, vector refit, or new H-label candidate.",
        statistical_unit="Two DEV sources; backgrounds, experts, dictionaries and baselines are repeated/nested conditions.",
        source_sha256={source.name:hash_file(source),prior.name:hash_file(prior)},
        source_payload_sha256=hashlib.sha256(payload(source)).hexdigest(),
        verification=errors,
        records=rows,geometry=geometry,aggregates=aggregates,
        direct_single_step_arithmetic=one_step)


if __name__=="__main__":
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=here/"D176_OBSERVABILITY.json.gz")
    ap.add_argument("--prior",type=Path,default=here/"D176_REPLAY.json")
    ap.add_argument("--output",type=Path,default=here/"observability_recomputed.json")
    args=ap.parse_args()
    out=run(args.source,args.prior)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(out["verification"],indent=2))
