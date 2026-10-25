"""Verify exact numerical reproduction of two fixed native E0 source repeats."""
import argparse
import json
from pathlib import Path

from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["dense","A","B","AB","dense_repeat"]


def verify(primary_root, cross_root):
    protocol_path = ROOT/"config/e0_finite_cross_worker_20261025.json"
    protocol = json.loads(protocol_path.read_text())
    result = {"status":"PASS_EXACT_CROSS_WORKER_NUMERICAL_REPRODUCTION",
              "scope":"Technical repeats of two existing sources on another A100 worker, same fixed groups/site/arms. No additional independent source, performance or quality acceptance.",
              "tool_sha256":sha256(Path(__file__)),"protocol_sha256":sha256(protocol_path),
              "independent_source_clusters_added":0,"sources":[],"generations_started":0}
    for index in protocol["source_indices"]:
        paths = [root/f"source{index:02d}_attempt1/RUN.json" for root in [primary_root,cross_root]]
        if sha256(paths[0])!=protocol["primary_reference_RUN_sha256"][str(index)]:
            raise ValueError("Frozen primary reference changed")
        reports = [json.loads(path.read_text()) for path in paths]
        primary,cross = reports
        if cross["protocol_sha256"]!=sha256(protocol_path):
            raise ValueError("Wrong cross-worker protocol")
        if cross["assigned_GPU_resources"]["violations"]:
            raise ValueError("Cross-worker GPU isolation failed")
        result["generations_started"] += cross["generations_started"]
        if any(r["status"]!="PASS_FIXED_E0_FINITE_NATIVE_DIAGNOSTIC" or list(r["phases"])!=ARMS for r in reports):
            raise ValueError("Require both completed native studies")
        if primary["assets"]!=cross["assets"] or primary["source_id"]!=cross["source_id"]:
            raise ValueError("Different frozen assets or input")
        for name,expected in protocol["driver_sources_sha256"].items():
            if sha256(paths[1].parent/"EXECUTED_SOURCES"/name)!=expected:
                raise ValueError("Replica executed source snapshot changed")
        counts = {"trajectory_tensor_hashes":0,"full_local_tensor_hashes":0,"prefix_hashes":0,"video_hashes":0}
        mismatches = []
        for arm in ARMS:
            a,b = [r["phases"][arm] for r in reports]
            if a["native_counts"]!=b["native_counts"]:
                raise ValueError("Native invocation counts differ")
            for label,fields in [("full_local_tensor_hashes","state_sha256"),("prefix_hashes","prefix")]:
                if set(a[fields])!=set(b[fields]):
                    raise ValueError("Different captured fields")
                for key in a[fields]:
                    counts[label]+=1
                    if a[fields][key]!=b[fields][key]:
                        mismatches.append(f"{arm}/{fields}/{key}")
            counts["video_hashes"]+=1
            if a["video_sha256"]!=b["video_sha256"]:
                mismatches.append(f"{arm}/video")
            for run,path in zip(reports,paths):
                if sha256(path.parent/f"{arm}.mp4")!=run["phases"][arm]["video_sha256"]:
                    raise ValueError("Captured video bytes changed")
            for step in range(4):
                for run,path in zip(reports,paths):
                    item=run["phases"][arm]["trajectory"][step]
                    if item["step"]!=step or sha256(path.parent/item["path"])!=item["sha256"]:
                        raise ValueError("Captured trajectory changed")
                for key in ["input_sha256","flow_sha256","post_sha256"]:
                    counts["trajectory_tensor_hashes"]+=1
                    if a["trajectory"][step][key]!=b["trajectory"][step][key]:
                        mismatches.append(f"{arm}/step{step}/{key}")
        hardware = {role:run["assigned_GPU_resources"]["samples"][0]["hardware"][0][:3]
                    for role,run in zip(["primary","replica"],reports)}
        result["sources"].append({"index":index,"primary_RUN_sha256":sha256(paths[0]),
            "replica_RUN_sha256":sha256(paths[1]),"checked_hash_counts":counts,
            "all_compared_hashes_match":not mismatches,"mismatches":mismatches,
            "hardware_name_driver_memory_MiB":hardware,"sampled_GPU_isolation_pass":True})
        if mismatches:
            result["status"]="NATIVE_CROSS_WORKER_DIFFERENCE_OBSERVED"
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-root",type=Path,required=True)
    p.add_argument("--cross-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve the first verification")
    write_json(args.out,verify(args.primary_root,args.cross_root))


if __name__=="__main__":
    main()
