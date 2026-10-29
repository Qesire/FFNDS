"""S2 native saved-source temporal statistics: draft or registered CPU execution.

Input: completed and audited ten-source original BF16 Dense 21tau packet corpus.
Output: source/step/block/tau-lag/frequency/effective-rank statistical evidence.
NO model/GPU/fitting/video; cannot execute until its new scope is registered
with exact protocol, hashes, budgets, and authorization in ACTIVE_RESEARCH.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"tools") not in sys.path:
    sys.path.insert(0,str(ROOT/"tools"))
import e0_raw_bf16_full21tau_analysis as data
import raw_bf16_temporal_statistics as stats

ID="E0_S2_FULL21TAU_TEMPORAL_STATISTICS_CPU_20261029"
PARENT="config/e0_raw_bf16_full21tau_new10_geometry_obs_cpu_20261029_v1.json"
OUTPUT="ffnds-derived/e0_s2_full21tau_temporal_statistics_cpu_20261029/attempt1/worker/results"
CPU={"GPU":0,"CPU_threads":4,"wall_seconds":7200,"outer_seconds":7500,
     "packet_bytes_max":32<<30,"output_bytes_max":256<<20,"automatic_retry":False}
METRICS=("V1","h","Y","gY","R","u","rowLN_u")
VIEWS=("all21","tau1_to20")
KEYS=("ac_energy_fraction","lag1_centered","lag2_centered","lag5_centered",
      "lag10_centered","low_band","mid_band","high_band","rank95",
      "participation_rank","difference_peak_ratio","direction_cosine")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def load_inputs(parent,hf):
    index,metadata=data.collect_inputs(parent,hf)
    need(index == parent["input_index"] and metadata == parent["metadata_inputs"],
         "Parent source/payload index or metadata changed")
    need(len(index["cases"])==10 and sum(len(c["packets"]) for c in index["cases"])==1600,
         "Ten distinct sources and all160 packets are required")
    return index


def draft(hf):
    parentpath=ROOT/PARENT
    parent=data.read(parentpath)
    index=load_inputs(parent,hf)
    count=sum(p["payload"]["bytes"] for c in index["cases"] for p in c["packets"])
    need(count < CPU["packet_bytes_max"],"Packet byte budget exceeded")
    code="tools/e0_s2_temporal_statistics_cpu.py"
    core="tools/raw_bf16_temporal_statistics.py"
    return {"id":ID,"status":"DRAFT_ONLY_NOT_AUTHORIZED",
            "parent_protocol":{"path":PARENT,"sha256":digest(parentpath)},
            "code_sha256":digest(ROOT/code),"statistics_sha256":digest(ROOT/core),
            "source_count":10,"packet_count":1600,"expected_packet_bytes":count,
            "resources":CPU,"operands":list(METRICS),"views":list(VIEWS),
            "source_state":"original_BF16_DENSE_TEACHER","replay_new_model_GPU_calls":0,
            "execution_requires":"EXACT_NEW_ACTIVE_SCOPE_AND_FROZEN_SHA",
            "motion_alignment":"NOT_OBSERVED_FIXED_XY_ONLY"}


def authorize(plan,sha):
    active=data.read(ROOT/"research/ACTIVE_RESEARCH.json")
    scopes=active["experiment_authority"]["current_user_authorized_scopes"]
    records=[x for x in scopes if x.get("id")==ID and x.get("protocol_sha256")==sha]
    need(len(records)==1 and records[0].get("max_gpu_cards")==0
         and records[0].get("execution_status") in ("AUTHORIZED_NOT_STARTED","RUNNING_FIXED_SCOPE"),
         "Scope is not independently registered/authorized; never run draft")
    need(plan["status"]=="FROZEN_REGISTERED" and plan["resources"]==CPU
         and plan["source_count"]==10 and plan["packet_count"]==1600,
         "Frozen registered protocol differs")


def extract_scalars(stat):
    lags={r["lag"]:r for r in stat["lag"]}
    sp=stat["spectrum"]["band_fractions"]
    out=(stat["AC_fraction_of_total_signal_energy"],
         *[lags[k]["centered_correlation"] if k in lags else None for k in (1,2,5,10)],
         sp["low_1_2"],sp["mid_3_6"],sp["high_7_plus"],
         stat["temporal_gram"]["rank95"],
         stat["temporal_gram"]["effective_rank_participation"],
         stat["events"]["first_difference_peak_to_mean"],
         stat["events"]["successive_difference_direction_cosine"])
    need(len(out)==len(KEYS),"Stat labels and values differ")
    return out


def run(args):
    protocol=args.protocol.resolve()
    plan=data.read(protocol); sha=digest(protocol)
    authorize(plan,sha)
    need(plan["parent_protocol"]=={"path":PARENT,"sha256":digest(ROOT/PARENT)}
         and plan["code_sha256"]==digest(ROOT/"tools/e0_s2_temporal_statistics_cpu.py")
         and plan["statistics_sha256"]==digest(ROOT/"tools/raw_bf16_temporal_statistics.py"),
         "Code or parent SHA changed; freeze a new protocol")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")==""
         and "torch" not in sys.modules and "numpy" not in sys.modules
         and len(os.sched_getaffinity(0))<=4,
         "New hidden-CUDA bounded CPU-only process required")
    from worker_identity import require_worker
    require_worker()
    out=(args.hf_root/OUTPUT).resolve()
    need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(),"Fresh output only")
    out.mkdir(parents=True)
    output=data.OutputBudget(out,CPU["output_bytes_max"])
    report={"id":ID,"status":"STARTED_NEW_S2_TEMPORAL_STATISTICS_CPU",
            "protocol_sha256":sha,"source_count":10,"packets":0,
            "original_u_neutrals":0,"actual_packet_bytes_read":0,
            "model_calls":0,"GPU_calls":0,"fits":0,"new_videos":0,
            "source_state":"DENSE_TEACHER","automatic_retry":False}
    started=time.monotonic()
    previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("S2 CPU timeout")))
    signal.alarm(CPU["wall_seconds"])
    def save():
        report["wall_seconds"]=time.monotonic()-started
        output.json("RUN.json",report,emergency=True)
    save()
    try:
        parent=data.read(ROOT/PARENT)
        index=load_inputs(parent,args.hf_root)
        need(plan["expected_packet_bytes"]==sum(p["payload"]["bytes"] for c in index["cases"] for p in c["packets"]),
             "Frozen input byte budget changed")
        for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[key]="4"
        import numpy as np
        need(np.__version__==parent["numpy_version"],"Parent and current NumPy versions differ")
        vectors={}
        all_rows=[]
        for case in index["cases"]:
            need(case["source_id"] not in vectors,"Duplicate image cluster")
            values=[]
            for entry in case["packets"]:
                raw,_=data.stream_payload(args.hf_root,entry["payload"],report)
                packet=data.decode_packet(raw,entry,data.core.rows(),np,data.SHAPES,
                                          whole_SHA_already_stream_verified=True)
                data.core.neutral_postwrite(packet,np)
                operands=data.core.operands(packet,np)
                detail={}
                flat=[]
                for operand in METRICS:
                    x=operands[operand].reshape(21,8,-1)
                    detail[operand]={}
                    for view in VIEWS:
                        trajectory=data.core.time_view(x,view)
                        item=stats.summarize(trajectory,np)
                        detail[operand][view]=item
                        flat.extend(np.nan if v is None else float(v)
                                    for v in extract_scalars(item))
                record={"source_id":case["source_id"],
                        "native_source_ordinal":case["native_source_ordinal"],
                        "step":entry["step"],"block":entry["block"],
                        "packet":entry["payload"],
                        "row_sha256":entry["tensor_metadata"]["rows"]["raw_tensor_sha256"],
                        "values":detail}
                output.append("POSITIONS.jsonl",(json.dumps(record,ensure_ascii=False,
                                                             separators=(",",":"),allow_nan=False)+"\n").encode())
                values.append((entry["step"],entry["block"],flat))
                report["packets"]+=1
                report["original_u_neutrals"]+=1
                report["last_packet"]=[case["native_source_ordinal"],entry["step"],entry["block"]]
                save()
                del packet,operands,detail,record,raw
            need(len(values)==160,"Every image must cover all160 sites")
            vectors[case["source_id"]]=values
        need(report["packets"]==report["original_u_neutrals"]==1600
             and report["actual_packet_bytes_read"]==plan["expected_packet_bytes"],
             "Full source packet byte/neutral budget not completed")
        labels=[f"{op}/{view}/{key}" for op in METRICS for view in VIEWS for key in KEYS]
        per_source=[]
        per_source_step=[]
        for source,rows in vectors.items():
            a=np.full((4,40,len(labels)),np.nan)
            for step,block,v in rows:
                a[step,block]=v
            need(np.isfinite(a).sum()>0,"No eligible temporal statistics")
            def mean(x,axis):
                count=np.isfinite(x).sum(axis=axis)
                total=np.where(np.isfinite(x),x,0).sum(axis=axis)
                out=np.full(total.shape,np.nan)
                np.divide(total,count,out=out,where=count>0)
                return out
            per_source.append({"source_id":source,
                               "all160":mean(a.reshape(160,-1),0),
                               "by_step":mean(a,1)})
            per_source_step.append(a)
        matrix=np.array([x["all160"] for x in per_source],dtype=float)
        valid=np.isfinite(matrix)
        count=valid.sum(axis=0)
        mean_source=np.full(count.shape,np.nan)
        np.divide(np.where(valid,matrix,0).sum(axis=0),count,
                  out=mean_source,where=count>0)
        def clean(x):
            if isinstance(x,np.ndarray): return clean(x.tolist())
            if isinstance(x,list): return [clean(y) for y in x]
            if isinstance(x,(np.floating,float)) and not np.isfinite(x): return None
            if isinstance(x,(np.integer,)): return int(x)
            return x
        report_summary={"schema":"ffnds.s2.temporal_stats_sources_equal.v1",
                        "id":ID,"protocol_sha256":sha,"source_count":10,
                        "labels":labels,"sources":[{"source_id":x["source_id"],
                        "all160":clean(x["all160"]),"by_step":clean(x["by_step"])}
                        for x in per_source],
                        "source_equal_all160":clean(mean_source),"eligible_sources":clean(count),
                        "unit":"independent image cluster; 160 nested sites per cluster",
                        "source_role":"new exposed development; no blind confirmation",
                        "motion_alignment":"NOT_MEASURED","student_free_rollout":"NOT_MEASURED"}
        output.json("SUMMARY.json",report_summary)
        report.update(status="COMPLETE_S2_TEMPORAL_STATISTICS_CPU_SOURCE_EQUAL",
                      result_files={"positions":str(out/"POSITIONS.jsonl"),
                                    "summary":str(out/"SUMMARY.json")})
    except BaseException as error:
        report.update(status="FAILED_S2_TEMPORAL_STATISTICS_CPU_PRESERVED_NO_RETRY",
                      error_type=type(error).__name__,error=str(error))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        try: save()
        finally: output.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("draft","run"))
    p.add_argument("--hf-root",type=Path,default=data.HF)
    p.add_argument("--protocol",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    a.hf_root=a.hf_root.resolve()
    if a.action=="draft":
        need(a.out and not a.out.exists(),"New draft output required")
        a.out.write_text(json.dumps(draft(a.hf_root),indent=2)+"\n")
        print("DRAFT_ONLY_NO_EXECUTION")
    else:
        need(a.protocol is not None,"Frozen registered protocol required")
        run(a)


if __name__=="__main__":
    main()
