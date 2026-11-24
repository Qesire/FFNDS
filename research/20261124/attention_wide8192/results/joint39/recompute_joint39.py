#!/usr/bin/env python3
"""Reanalyse published joint39 sufficient statistics; no model or GPU execution.

Example:
  python recompute_joint39.py --source-dir /path/to/joint39 \
    --panel-index /path/to/attention_panel_index.json --output joint39_recomputed.json

RESULT_0392.json supplies scalar energies and signed Gram matrices.
VECTOR_AUDIT_0392.json supplies independently audited H448 least-squares diagnostics.
The local script checks and recomputes these aggregates, not the school safetensors.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np

EVIDENCE_COMMIT = "b93ac8ee2ed54ccc7cb5b7cbe635fd5e0399e94b"

def gram_analysis(record):
    g = np.asarray(record["gram"], dtype=np.float64)
    d = np.diag(g)
    total = float(record["B11_error"])
    pair = {k: float(2*g[i,j]) for k,i,j in (
        ("two_A_F",0,1),("two_A_I",0,2),("two_F_I",1,2))}
    cos = g / np.sqrt(d[:,None]*d[None,:])
    a_given_f = float(d[0]+d[2]+2*g[0,2])
    f_given_a = float(d[1]+d[2]+2*g[1,2])
    return {
        "terms": record["terms"], "energies": d.tolist(), "total":total,
        "gram_sum_error":float(g.sum()-total),
        "gram_eigenvalues":np.linalg.eigvalsh(g).tolist(),
        "cosines":cos.tolist(),
        "signed_double_cross_terms":pair,
        "signed_cross_sum":float(g.sum()-d.sum()),
        "total_over_three_diagonal_energies":float(total/d.sum()),
        "total_over_A_plus_F_energies":float(total/(d[0]+d[1])),
        "interaction_energy_over_A":float(d[2]/d[0]),
        "interaction_energy_over_F":float(d[2]/d[1]),
        "A_F_cross_added_energy":float(2*g[0,1]),
        "interaction_net_added_to_A_plus_F_vector":float(d[2]+2*g[0,2]+2*g[1,2]),
        "attention_effect_with_approx_F_energy":a_given_f,
        "attention_effect_with_approx_F_over_native_F":float(a_given_f/d[0]),
        "attention_effect_with_approx_F_cos_native_F":float((d[0]+g[0,2])/math.sqrt(d[0]*a_given_f)),
        "FFN_effect_with_approx_A_energy":f_given_a,
        "FFN_effect_with_approx_A_over_native_A":float(f_given_a/d[1]),
        "F_over_A_energy":float(d[1]/d[0]),
        "edge_signed_cross_sums": {
            kind:float(e["total"]-e["E_A"]-e["E_F"]-e["I_AF"])
            for kind,e in record["edge_energies"].items()
        }
    }

def pair_parts(score):
    return {
        "point":float(score["point"]),
        "spatial_common":float(score["point"]-score["spatial"]/2),
        "spatial_difference":float(score["spatial"]/2),
        "temporal_common":float(score["point"]-score["temporal"]/2),
        "temporal_difference":float(score["temporal"]/2)
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir",type=Path,required=True)
    ap.add_argument("--panel-index",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    paths={name:args.source_dir/name for name in (
        "RESULT_0392.json","VECTOR_AUDIT_0392.json","P64_AUDIT.json","STATUS.json")}
    contents={k:json.loads(p.read_text()) for k,p in paths.items()}
    result=contents["RESULT_0392.json"]
    vector=contents["VECTOR_AUDIT_0392.json"]
    diagnostics={(v["background"],v["expert"]):v for v in vector["vectors"]}
    panel_check={"status":"assumption_from_previously_verified_matching",
        "identity":"E_difference=E_edge/2; E_common=E_point-E_edge/2"}
    if args.panel_index:
        panel=json.loads(args.panel_index.read_text())
        check={}
        for key in ("spatial","temporal"):
            c=Counter(i for pair in panel[key+"_edges_rows"] for i in pair)
            assert set(c)==set(panel["held_rows"]) and set(c.values())=={1}
            check[key]={"edges":len(panel[key+"_edges_rows"]),"held_nodes":len(c),"degree":1}
        panel_check.update(status="verified_from_panel_index",checks=check,
            sha256=hashlib.sha256(args.panel_index.read_bytes()).hexdigest())
    out={
        "status":"EXPLORATORY_RECOMPUTATION_OF_PUBLISHED_REAL_MODEL_STATISTICS",
        "evidence_commit":EVIDENCE_COMMIT,
        "new_model_executions":0,
        "raw_safetensors_recomputed_locally":False,
        "verified_joint_sources":["vbi_0392"],
        "independent_source_count":1,
        "nested_sites":4,
        "second_source_status":contents["STATUS.json"]["second_source"],
        "input_sha256":{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        "edge_partition":panel_check,
        "scopes":{
            "block_interaction":"Fixed-reader four-arm finite interaction in the residual block.",
            "flow_interaction":"Four-arm interaction after nonlinear head; it includes head composition.",
            "head_relative_sensitivity":"Ratio of F/A energies within flow divided by F/A energies within block. This is a finite, direction-specific comparison, not an operator norm.",
            "P_H_gap":"Published scalar P64 risk minus the H448 least-squares projection floor; represents coefficient-selection/extension loss within this fixed feature family, with FP32 output cast residual. Not pure sampling causality.",
            "head_replay_error":"Published baseline local-head versus full-head discrepancy. Not a universal noise floor for each candidate or an identical teacher-reference shift."
        },
        "sites":{}
    }
    for bg,b in result["results"].items():
        for expert,r in b["experts"].items():
            diag=diagnostics[bg,expert]
            assert diag["asset_sha256"]==r["asset"]["sha256"]
            scores=r["flow_scores"]
            j=scores["joint_flow"]["point"]
            i=scores["independent_flow"]["point"]
            cheap=scores["cheap"]["point"]
            floor=diag["diagnostic"]["H448_oracle_floor_point"]
            all_h=diag["diagnostic"]["H448_same_ridge_point"]
            block=gram_analysis(r["branch_gram"]["block_exit"])
            flow=gram_analysis(r["branch_gram"]["flow"])
            assert abs(block["gram_sum_error"])<1e-8 and abs(flow["gram_sum_error"])<1e-10
            p=r["prefix_response"]
            parts={name:pair_parts(scores[name]) for name in (
                "B10","B01","B11_fixed","B11_refit","cheap","joint_flow")}
            pair_ratios={cmp:{metric:parts[left][metric]/parts[right][metric]
                for metric in parts[left]} for cmp,left,right in (
                    ("joint_over_independent","joint_flow","B11_fixed"),
                    ("joint_over_cheap","joint_flow","cheap"),
                    ("independent_over_cheap","B11_fixed","cheap"))}
            s={
                "source":b["source"],"background":bg,"expert":expert,
                "step":r["step"],"attention_arm":r["attention_arm"],
                "flow_scores":scores,"block_gram":block,"flow_gram":flow,
                "same_residual_space_attention_transport":{
                    "R_error_energy":p["R"],"T_error_energy":p["T"],
                    "block_E_A_energy":block["energies"][0],
                    "cross_T_over_R_energy":p["T"]/p["R"],
                    "native_FFN_block_E_A_over_T_energy":block["energies"][0]/p["T"],
                    "native_FFN_block_E_A_over_T_norm":math.sqrt(block["energies"][0]/p["T"]),
                    "native_tail_block_E_A_over_R_energy":block["energies"][0]/p["R"],
                    "note":"Cross norm preservation alone does not determine directional rotation. U uses normalization/modulation coordinates, so raw U/T is not reported as a residual-space gain."
                },
                "head_relative_F_vs_A_sensitivity":flow["F_over_A_energy"]/block["F_over_A_energy"],
                "joint_error_budget":{
                    "P64_joint":j,"H448_same_ridge":all_h,"H448_OLS_floor":floor,
                    "independent":i,"cheap":cheap,
                    "P64_minus_floor":j-floor,
                    "P64_minus_H448_same_ridge":j-all_h,
                    "H448_ridge_minus_floor":all_h-floor,
                    "floor_minus_independent":floor-i,
                    "P64_minus_independent":j-i,
                    "P64_over_independent":j/i,
                    "floor_over_independent":floor/i,
                    "P64_over_floor":j/floor,
                    "P64_over_cheap":j/cheap,
                    "floor_over_cheap":floor/cheap,
                    "independent_over_cheap":i/cheap,
                    "floor_fraction_of_P64_error":floor/j,
                    "P64_minus_floor_fraction":(j-floor)/j,
                    "P64_rank":diag["diagnostic"]["P64_rank"],
                    "P64_condition":diag["diagnostic"]["P64_condition"],
                    "same_ridge_over_floor":all_h/floor
                },
                "fixed_edge_error_parts":parts,
                "fixed_edge_error_part_ratios":pair_ratios,
                "refit":{
                    "delta_energy":r["feedback_refit_delta_energy"],
                    "flow_error_ratio":{metric:scores["B11_refit"][metric]/scores["B11_fixed"][metric] for metric in ("point","spatial","temporal")},
                    "delta_norm_over_fixed_error_norm":math.sqrt(r["feedback_refit_delta_energy"]/i),
                    "cos_delta_with_fixed_error":float((scores["B11_refit"]["point"]-i-r["feedback_refit_delta_energy"])/(2*math.sqrt(i*r["feedback_refit_delta_energy"])))
                },
                "recorded_replay_numerics":{
                    "head_relative_l2":r["live_flow_panel_replay_relative"],
                    "head_baseline_point_discrepancy":r["live_flow_panel_error"]["point"],
                    "block_relative_l2":r["real_block_error_relative"],
                    "block_error_energy":r["real_block_error_energy"]
                }
            }
            out["sites"][bg+"/"+expert]=s
    cross={}
    for expert in ("high","low"):
        d=out["sites"]["Dense/"+expert];w=out["sites"]["wide8192/"+expert]
        cross[expert]={
            "wide_over_Dense":{
                "cheap":w["joint_error_budget"]["cheap"]/d["joint_error_budget"]["cheap"],
                "joint":w["joint_error_budget"]["P64_joint"]/d["joint_error_budget"]["P64_joint"],
                "floor":w["joint_error_budget"]["H448_OLS_floor"]/d["joint_error_budget"]["H448_OLS_floor"],
                "independent":w["joint_error_budget"]["independent"]/d["joint_error_budget"]["independent"],
                "attention_only_flow":w["flow_gram"]["energies"][0]/d["flow_gram"]["energies"][0],
                "FFN_only_flow":w["flow_gram"]["energies"][1]/d["flow_gram"]["energies"][1]
            },
            "scope":"Same source/expert, distinct background current states and anchors; this is a paired background association, not an isolated downstream-module intervention."
        }
    out["cross_background"]=cross
    out["direct_scheduler_weighted_flow_energy"]={}
    for bg in ("Dense","wide8192"):
        high=out["sites"][bg+"/high"];low=out["sites"][bg+"/low"]
        h_high,h_low=float(Fraction(5,48)),float(Fraction(5,8))
        for arm in ("B10","B01","B11_fixed","cheap","joint_flow"):
            eh=high["flow_scores"][arm]["point"]*h_high*h_high
            el=low["flow_scores"][arm]["point"]*h_low*h_low
            out["direct_scheduler_weighted_flow_energy"][bg+"/"+arm]={
                "high":eh,"low":el,"low_over_high":el/eh,
                "scope":"Direct one-step latent injection energy on measured local fork; high/low contributions are not summed or treated as independent final-trajectory errors."}
    probe=contents["P64_AUDIT.json"]
    out["probe_checks"]={
        "count":len(probe["rows"]),
        "sources":sorted(set(r["source"] for r in probe["rows"])),
        "min_relative_l2":min(r["relative_l2"] for r in probe["rows"]),
        "max_relative_l2":max(r["relative_l2"] for r in probe["rows"]),
        "scope":"Subquery numerical check covers two sources; full joint four-arm evidence covers one source."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    for name,s in out["sites"].items():
        z=s["joint_error_budget"]
        print(name,json.dumps({
            "joint_over_independent":z["P64_over_independent"],
            "floor_over_independent":z["floor_over_independent"],
            "joint_over_cheap":z["P64_over_cheap"],
            "T_to_B_attention_energy":s["same_residual_space_attention_transport"]["native_FFN_block_E_A_over_T_energy"],
            "head_relative_F_sensitivity":s["head_relative_F_vs_A_sensitivity"],
            "flow_cos_A_F":s["flow_gram"]["cosines"][0][1],
            "interaction_energy_over_A":s["flow_gram"]["interaction_energy_over_A"]}))
    print(json.dumps({"output":str(args.output),"new_model_executions":0}))

if __name__=="__main__":
    main()

