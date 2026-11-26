"""Recompute historical strict FFN comparisons from source-extracted records.
No GPU inference. Original 2.6 MB Git blob identity is recorded in the input.
"""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT=Path(__file__).resolve().parent
x=json.loads((ROOT/"STRICT_NUMERIC_INPUT.json").read_text())
rows=x["scores"]
assert len(rows)==495
keys={(r["source"],r["arm"],r["dimension"]) for r in rows}
assert len(keys)==495
sources=sorted({r["source"] for r in rows})
arms=sorted({r["arm"] for r in rows})
assert len(sources)==11 and len(arms)==5
by={(r["source"],r["arm"],r["dimension"]):r["value"] for r in rows}
dimensions=sorted({r["dimension"] for r in rows})
summaries=[]
for arm in arms:
    for dimension in dimensions:
        vals=[by[(s,arm,dimension)] for s in sources]
        diffs=[by[(s,arm,dimension)]-by[(s,"Dense",dimension)] for s in sources]
        summaries.append(dict(arm=arm,dimension=dimension,mean=mean(vals),
                              mean_delta_vs_dense=mean(diffs),
                              min_delta=min(diffs),max_delta=max(diffs),
                              sources_below_dense=sum(d<0 for d in diffs)))
for r in x["reference_summaries"]:
    if r["dimension"]=="camera_motion":
        continue
    a=next(v for v in summaries if v["arm"]==r["arm"] and v["dimension"]==r["dimension"])
    assert abs(a["mean_delta_vs_dense"]-r["cluster_equal_mean_delta"])<1e-12
motion=[]
for arm in arms:
    rr=[r for r in rows if r["arm"]==arm and r["dimension"]=="dynamic_degree"]
    assert len(rr)==11
    motion.append(dict(arm=arm,continuous_raft_mean=mean(r["continuous_motion"] for r in rr)))
costs=[]
for arm in arms:
    rr=[r for r in x["costs"] if r["arm"]==arm and not r.get("ablation")]
    assert all(r["uninstrumented"] is False for r in rr)
    costs.append(dict(arm=arm,n=len(rr),observed_wall_median=median(r["wall_seconds"] for r in rr),
                      write_gib_median=median(r.get("cache_io",{}).get("write",{}).get("bytes",0)/2**30 for r in rr),
                      read_gib_median=median(r.get("cache_io",{}).get("read",{}).get("bytes",0)/2**30 for r in rr),
                      wall_scope=sorted({r["wall_scope"] for r in rr})))

# Static MAC comparison using the exact measured selected-channel counts.
# Chip port: 8 fixed dense + 76 anchors + 76 hits; 247 complete 128-token blocks,
# 52 native tail tokens per hit. Full MLP costs two native projections.
N,d,m,k,r=31668,5120,13824,2560,1408
full=4*N*d*m
chip=[]
for z in x["costs"]:
    if z["arm"]!="Chipmunk Author Safe" or z.get("ablation"):
        continue
    assert z["chip_selected_blocks"]==76*247
    assert z["ffn_sites"]=={"fixed_dense":8,"anchor":76,"hit":76}
    selected_flops=4*128*d*z["chip_selected_columns_sum"]
    tail_flops=76*4*52*d*m
    proxy_flops=76*2*247*d*m
    ratio=(84*full+selected_flops+tail_flops+proxy_flops)/(160*full)
    chip.append(dict(source=z["source"],
                     mean_selected_channels=z["chip_selected_columns_sum"]/z["chip_selected_blocks"],
                     principal_ffn_ratio=ratio))
ratios=dict(
    current_wide8192_major_ffn_cycle=(1+(2*N*d*d+4*N*d*k)/full)/2,
    strict_icc4s_r1408_major_ffn_cycle=(1+4*N*r*(d+m)/full)/2,
    chip_port_major_ffn_cycle_min=min(a["principal_ffn_ratio"] for a in chip),
    chip_port_major_ffn_cycle_max=max(a["principal_ffn_ratio"] for a in chip),
)
out=dict(source=x["source"],scope=x["scope"],
         numerical_checks="495 unique metric rows, 11 sources, 5 arms; means match source summaries to 1e-12",
         quality=summaries,motion=motion,observed_costs=costs,
         static_ffn_ratios=ratios,chip_static_per_source=chip,
         static_scope="Includes full anchor/hit lifecycle, dense first-two layers and valid dense tail, selected GEMMs and head block-mean W1 proxy; excludes nonlinearities, packing/index-copy, statistics, storage and memory traffic. Current-wide8192 line excludes its small P64/readout GEMMs. Different historical batches: no current-vs-Chip performance win is inferred.",
         timing_scope="Historical diagnostic and disk-backed cache timings, not uninstrumented kernel or steady-state system speed.")
(ROOT/"STRICT_COMPARISON_VERIFIED.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(dict(static_ffn_ratios=ratios,motion=motion,observed_costs=costs),ensure_ascii=False,indent=2))

