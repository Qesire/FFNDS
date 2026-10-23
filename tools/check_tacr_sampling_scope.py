"""Source-aware TACR v2 sampling / historic execution status guard. No model/GPU use."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))
def replay():
    src=read("research/20261023/R17_fresh23_input_audit.json")
    gen=read("research/20261023/R17E0_partial_generations.json")
    inter=read("research/20261023/R17E0_interruption_audit.json")
    r=read("research/20261023/R17E0_time_allocation_readout.json")
    names={"dense_DQ","UB1","fixed_stride","fixed_hash","dynamic_residual"}
    grouped={}
    for x in gen["completed"]:
        j=x["job"]
        if j.get("phase")=="quality":
            grouped.setdefault(j["source"],[]).append(j["arm"])
    complete=sorted(i for i,rs in grouped.items() if set(rs)==names and len(rs)==5)
    partial=sorted(i for i in grouped if i not in complete)
    audit={"source_count":src["source_count"],"planned_strata":src["strata"],
           "completed_generation_count":len(gen["completed"]),
           "quality_clip_count":sum(len(x) for x in grouped.values()),
           "quality_complete_groups":[{"source_index":n,"source_group":src["inputs"][n]["source_group"],"stratum":src["inputs"][n]["stratum"],"image_type":src["inputs"][n]["image_type"]} for n in complete],
           "quality_partial_groups":[{"source_index":n,"source_group":src["inputs"][n]["source_group"],"arms":sorted(grouped[n]),"stratum":src["inputs"][n]["stratum"]} for n in partial],
           "warmups":inter["completed_warmups"],"timing_repetitions":inter["completed_timing_generations"],
           "official_metric_calls":inter["actual_debits"]["metric_evaluator_calls"],
           "latent_layout":r["native_layout"],"first_time_bin_dense_rate":r["first_latent_time_dense_fraction_mean"],
           "later20_dense_rate":r["other20_latent_time_dense_fraction_mean"],
           "adjacent_bin_mask_flip_rate":r["mean_adjacent_latent_time_flip_fraction"]["dynamic_residual"],
           "quality_exposure":"FIVE_PAIRED_SUBJECT_ONLY_SIXTH_PARTIAL","new_model_calls":0}
    return audit
def verify(plan):
    e=[]
    m=plan["model"]
    if m["latent_grid"]!=[21,29,52] or m["rows"]!=21*29*52:e.append("wrong native temporal/spatial grid")
    if m["groups"]*m["group_width"]!=m["hidden_channels"]:e.append("invalid native hidden channel groups")
    if m["UB1_step_base_groups"]!=[54,108,108,162]:e.append("original step-specific UB1 group budget changed")
    sp=plan["sampling"]
    if set(sp["step_ids"])!={0,1,2,3}:e.append("not all four denoising steps")
    if sp["conditioned_time_bins"]!=[0] or sp["remaining_time_bins"]!=list(range(1,21)):e.append("mistaken decoded-frame/latent time bin axes")
    if set(sp["source_strata"])!={"i2v_subject","i2v_background","camera_motion"}:e.append("missing source exposure strata")
    if sp["pilot_blocks"]!=[sp["depth"][k][0] for k in ("early","middle","late")]:e.append("bad pilot depth sampling")
    if len(set(sum(sp["depth"].values(),[])))!=6 or any(not all(lo<=x<=hi for x in sp["depth"][k]) for k,lo,hi in (("early",0,12),("middle",13,26),("late",27,39))):e.append("depth bins overlap/escape")
    if not plan.get("no_new_GPU_authority") or not plan.get("previous_trials_unchanged"):e.append("historic method eligibility corrupted")
    return e
def run():
    p=read("research/20261023/TACR_sampling_scope_v2.json")
    actual=replay()
    errors=verify(p)
    if actual!=read("research/20261023/TACR_sampling_evidence_v2.json"):errors.append("frozen R17 sampling evidence mismatches native receipts")
    if actual["completed_generation_count"]!=57 or actual["quality_clip_count"]!=27 or len(actual["quality_complete_groups"])!=5:errors.append("R17 interrupted sampling scope changed")
    if any(g["stratum"]!="i2v_subject" for g in actual["quality_complete_groups"]):errors.append("R17 completed strata misreported")
    if actual["official_metric_calls"]!=0:errors.append("R17 official metric execution scope changed")
    return errors,actual
if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--check",action="store_true")
    a.parse_args()
    errors,actual=run()
    print(json.dumps({"errors":errors,"quality_sources":len(actual["quality_complete_groups"]),"video_clips":actual["quality_clip_count"],"official_metrics":actual["official_metric_calls"],"new_model_calls":0}))
    raise SystemExit(bool(errors))
