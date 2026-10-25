"""Authorized full-W2 saved-operand GPU mechanism study, no model generation."""
import argparse
import json
from pathlib import Path
import shutil
import time

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json
from e0_intrinsic_math import (coefficient_scores, energy, group_values,
                               projection_description, source_fixed_score)

ROOT = Path(__file__).resolve().parents[1]


def execute(args):
    protocol = json.loads(args.protocol.read_text())
    for name, expected in protocol["driver_sources_sha256"].items():
        if sha256(ROOT/name) != expected:
            raise ValueError("Frozen analysis source changed")
    mp = ROOT/protocol["accepted_observations"]
    cp = ROOT/protocol["cache_manifest"]
    if sha256(mp) != protocol["accepted_observations_sha256"] or sha256(cp) != protocol["cache_manifest_sha256"]:
        raise ValueError("Source manifest identity changed")
    from worker_identity import require_worker
    require_worker()
    import torch
    from startup_gpu_reservation import reserve_before_assets
    if not args.execute_gpu or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("Explicit GPU flag and frozen A100 required")
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out/"EXECUTED_SOURCES"
    snapshot.mkdir()
    for name in protocol["driver_sources_sha256"]:
        target = snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    shutil.copyfile(args.protocol,snapshot/args.protocol.name)
    record = {"status":"STARTED_NOT_ACCEPTED","protocol_sha256":sha256(args.protocol),
              "new_model_generations":0,"new_native_FFN_calls":0,"new_native_norm_calls":0,
              "new_weights_or_training":0,"cases":[],"weights":[],
              "diagnostic_small_W2_subset_projections":0,
              "diagnostic_full_reference_projections":0,
              "diagnostic_batched_group_projections":0,
              "quality_or_speed_acceptance":False,"automatic_retry":False}
    write_json(args.out/"RUN.json",record)
    monitor = None
    started = time.perf_counter()
    try:
        reservation, monitor, receipt = reserve_before_assets(torch)
        record["CUDA_reservation"] = receipt
        record["environment"] = {"device":torch.cuda.get_device_name(0),"torch":torch.__version__,
                                 "CUDA":torch.version.cuda,"math_dtype":"float64",
                                 "native_weight_cast":"bfloat16"}
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest, cache = json.loads(mp.read_text()),json.loads(cp.read_text())
        runs = {}
        for source in manifest["sources"]:
            path = args.raw_root/f"source{source['index']:02d}_attempt1/RUN.json"
            if sha256(path) != source["run_sha256"]:
                raise ValueError("Accepted saved run changed")
            run = json.loads(path.read_text())
            if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
                raise ValueError("Require accepted collection")
            runs[source["index"]] = (path,run)

        def saved(index,step,block):
            run_path,run = runs[index]
            scope = next(s for s in run["scopes"] if (s["step"],s["block"])==(step,block))
            path = run_path.parent/"observed"/scope["path"]
            wp = run_path.parent/"observed"/scope["weight_binding"]["path"]
            if sha256(path) != scope["sha256"] or sha256(wp) != scope["weight_binding"]["sha256"]:
                raise ValueError("Saved tensor/panel changed")
            a,panel = load_arrays(path),load_arrays(wp)
            ids = [scope["row_plan"]["rows"].index(row) for row in scope["row_plan"]["uniform_rows"]]
            if len(ids)!=32:
                raise ValueError("Changed fixed row panel")
            selected = torch.tensor([g*64+k for g in scope["weight_binding"]["groups"] for k in range(64)])
            if not torch.equal(original_weight.index_select(1,selected).float(),
                               torch.from_numpy(panel["W2_panel"]).float()):
                raise ValueError("Full W2 does not reproduce all five saved panels")
            def device(value):return torch.as_tensor(value).to(device="cuda",dtype=torch.float64)
            return {"q":device(a["q2_post"][ids]),"gate":device(a["gate"]),
                    "bias":device(a["b2"]).reshape(1,-1),"native_y":device(a["ffn_output"][ids]),
                    "payload_sha256":scope["sha256"],"weight_panel_sha256":scope["weight_binding"]["sha256"],
                    "tensor_shapes":{k:list(v.shape) for k,v in a.items()}}

        with torch.inference_mode():
            for block in protocol["blocks"]:
                for expert,steps in [("high_noise_model",[0,1]),("low_noise_model",[2,3])]:
                    monitor.check()
                    shard = next(s for s in cache["shards"] if s["expert"]==expert and s["block"]==block)
                    sp = Path(shard["path"])
                    if sha256(sp) != shard["sha256"]:
                        raise ValueError("Original full block shard checksum failed")
                    payload = torch.load(sp,map_location="cpu",weights_only=False)
                    original_weight = payload[expert][f"blocks.{block}.ffn.2"]["weight_qdq"]
                    if tuple(original_weight.shape)!=(5120,13824):
                        raise ValueError("Incorrect full hidden/value dimensions")
                    w = original_weight.to(device="cuda",dtype=torch.bfloat16).double()
                    record["weights"].append({"expert":expert,"block":block,"shard_sha256":shard["sha256"],
                                              "original_weight_dtype":str(original_weight.dtype),
                                              "full_weight_shape":list(w.shape)})
                    for step in steps:
                        cases = {i:saved(i,step,block) for i in protocol["source_indices"]}
                        for mode in (["absolute","same_expert_innovation"] if step in [1,3] else ["absolute"]):
                            current = {i:{**v} for i,v in cases.items()}
                            if mode != "absolute":
                                for i,c in current.items():
                                    previous=saved(i,step-1,block)
                                    if not torch.equal(previous["bias"],c["bias"]):
                                        raise ValueError("Same-expert original bias changed")
                                    c["q"]=c["q"]-previous["q"]
                                    c["previous_payload_sha256"]=previous["payload_sha256"]
                            ns,gs = {},{}
                            for i,c in current.items():
                                ns[i]=coefficient_scores(c["q"],w,c["gate"])["value_diagonal"]
                                values=group_values(c["q"],w,c["gate"])
                                gs[i]=(values*values).sum(-1)
                                record["diagnostic_batched_group_projections"]+=1
                            for i,c in current.items():
                                monitor.check()
                                description=projection_description(c["q"],w,c["gate"],
                                    source_fixed_score(ns,i),source_fixed_score(gs,i),
                                    protocol["retained_fractions"],protocol["random_seed"]+100*block+step)
                                record["diagnostic_batched_group_projections"]+=1
                                record["diagnostic_small_W2_subset_projections"]+=4*len(protocol["retained_fractions"])
                                item={"source_index":i,"step":step,"block":block,"expert":expert,"mode":mode,
                                      **{k:c[k] for k in ["payload_sha256","weight_panel_sha256"]},
                                      "description":description}
                                if mode=="absolute":
                                    continuous=(c["q"]@w.T+c["bias"])*c["gate"]
                                    reference=c["native_y"]*c["gate"]
                                    item["continuous_vs_saved_native_relative_SSE"]=energy(continuous-reference)/energy(reference)
                                    item["tensor_shapes"]=c["tensor_shapes"]
                                    record["diagnostic_full_reference_projections"]+=1
                                else:
                                    item["previous_payload_sha256"]=c["previous_payload_sha256"]
                                record["cases"].append(item)
                                if len(record["cases"])>protocol["max_cases"]:
                                    raise RuntimeError("Diagnostic scope cap reached")
                            write_json(args.out/"RUN.json",record)
                            print(f"intrinsic saved {expert} block{block:02d} step{step} {mode}: six sources",flush=True)
                        del cases,current,ns,gs,values
                    del w,original_weight,payload
                    torch.cuda.empty_cache()
        if len(record["cases"])!=180:
            raise RuntimeError("Incomplete absolute/innovation scope")
        record.update(status="PASS_FULL_HIDDEN_SAVED_GPU_MECHANISM_DESCRIPTION",
                      elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        record.update(status="FAILED_PRESERVED",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if monitor is not None:
            monitor.close();record["assigned_GPU_resources"]=monitor.report()
        write_json(args.out/"RUN.json",record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--raw-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--execute-gpu",action="store_true")
    execute(p.parse_args())


if __name__=="__main__":main()
