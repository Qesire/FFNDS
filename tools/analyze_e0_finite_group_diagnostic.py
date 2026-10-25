"""Verify fixed native E0 controls and measure freely running four-arm responses."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["dense", "A", "B", "AB", "dense_repeat"]


def decoded_video_distances(folder):
    import imageio.v2 as imageio
    arms = ["dense", "A", "B", "AB"]
    readers = [imageio.get_reader(folder/f"{arm}.mp4",format="ffmpeg",input_params=["-threads","1"])
               for arm in arms]
    frame_errors = {a: [] for a in arms[1:]}
    shape = None
    try:
        for frames in itertools.zip_longest(*(iter(r) for r in readers)):
            if any(f is None for f in frames):
                raise ValueError("Unequal complete video frame counts")
            if len({f.shape for f in frames})!=1:
                raise ValueError("Unequal native video dimensions")
            shape = list(frames[0].shape)
            truth = frames[0].astype(np.float32)/255
            for arm,frame in zip(arms[1:],frames[1:]):
                d = frame.astype(np.float32)/255-truth
                frame_errors[arm].append(float(np.mean(d*d,dtype=np.float64)))
    finally:
        for reader in readers:
            reader.close()
    if any(len(v)!=81 for v in frame_errors.values()):
        raise ValueError("Expected81 complete decoded frames")
    return {"scope":"Decoded RGB difference to dense D-Q, not semantic/perceptual quality acceptance",
            "frames":81,"frame_shape":shape,
            "arms":{arm:{"mean_RGB_MSE":float(np.mean(values)),"max_frame_RGB_MSE":max(values),
                         "per_frame_RGB_MSE":values} for arm,values in frame_errors.items()}}


def squared_response(value, reference):
    d = value.astype(np.float64)-reference.astype(np.float64)
    norm = float(np.sum(reference.astype(np.float64)**2))
    error = float(np.sum(d*d))
    return {"SSE": error, "reference_energy": norm, "relative_SSE": error/norm}


def four_arm(dense, a, b, ab):
    a, b, ab = [v.astype(np.float64)-dense.astype(np.float64) for v in [a,b,ab]]
    curve = ab-a-b
    sse = [float(np.sum(v*v)) for v in [a,b,ab]]
    cross = float(2*np.sum(a*b))
    nonlinear = float(2*np.sum((a+b)*curve)+np.sum(curve*curve))
    remainder = sse[2]-sse[0]-sse[1]-cross-nonlinear
    if abs(remainder) > 1e-10*max(1.,*sse):
        raise ValueError("Four-arm response identity failed")
    return {"A_SSE": sse[0], "B_SSE": sse[1], "AB_SSE": sse[2],
            "AB_over_A_plus_B_SSE": sse[2]/(sse[0]+sse[1]),
            "linear_cross_term": cross, "nonlinear_term": nonlinear,
            "curvature_norm2": float(np.sum(curve*curve)),
            "additive_response_norm2": float(np.sum((a+b)**2)), "identity_residual": remainder}


def sample_directions(source, folder, observation_root):
    import torch
    old_folder = observation_root/f"source{source['index']:02d}_attempt1"
    run = json.loads((old_folder/"RUN.json").read_text())
    if sha256(old_folder/"RUN.json") != source["run_sha256"]:
        raise ValueError("Original accepted run changed")
    scope = next(s for s in run["scopes"] if (s["step"],s["block"])==(0,0))
    payload, weight = [old_folder/"observed"/p for p in [scope["path"],scope["weight_binding"]["path"]]]
    if sha256(payload)!=scope["sha256"] or sha256(weight)!=scope["weight_binding"]["sha256"]:
        raise ValueError("Original signed operands changed")
    a,w = load_arrays(payload),load_arrays(weight)
    plan = scope["row_plan"]
    ids = [plan["rows"].index(r) for r in plan["uniform_rows"]]
    samples = {arm: load_arrays(folder/f"{arm}_samples.safetensors") for arm in ARMS}
    if not np.array_equal(samples["dense"]["post"],a["residual_post_ffn"][ids]) or not np.array_equal(
        samples["dense"]["ffn_output"],a["ffn_output"][ids]):
        raise ValueError("Passive and native diagnostic dense samples differ")
    delta = {}
    for arm,groups in [("A",[0]),("B",[53]),("AB",[0,53])]:
        parts = []
        for g in groups:
            k = scope["weight_binding"]["groups"].index(g)
            parts.append((a["q2_post"][ids,g*64:(g+1)*64].astype(np.float64) @
                          w["W2_panel"][:,k*64:(k+1)*64].astype(np.float64).T)*a["gate"])
        delta[arm] = sum(parts)
    result = {"rows": len(ids), "native_dense_samples_match_original_passive": True, "arms": {}}
    # The frozen post_process uses two BF16 operations: multiply, then add.
    pre = torch.from_numpy(a["residual_pre_ffn"][ids]).bfloat16()
    gate = torch.from_numpy(a["gate"]).bfloat16()
    products = {}
    cast_norms = {}
    for arm in ARMS:
        product = torch.from_numpy(samples[arm]["ffn_output"]).bfloat16()*gate
        rebuilt = (pre+product).float().numpy()
        if not np.array_equal(rebuilt,samples[arm]["post"]):
            raise ValueError("Saved BF16 multiply/add does not reconstruct native writeback")
        products[arm] = product.float().numpy().astype(np.float64)
        state = samples[arm]["post"].astype(np.float64)
        centered = state-state.mean(axis=1,keepdims=True)
        normalized = centered/np.sqrt(np.mean(centered**2,axis=1,keepdims=True)+1e-6)
        cast_norms[arm] = torch.from_numpy(normalized).bfloat16().float().numpy().astype(np.float64)
    result["native_BF16_writeback_reconstructed_all_five_arms"] = True
    result["ideal_LN_then_BF16_value_match_fraction"] = {
        arm:float(np.mean(cast_norms[arm]==samples[arm]["nextnorm"])) for arm in ARMS}
    for arm in ["A","B","AB"]:
        actual = samples["dense"]["post"].astype(np.float64)-samples[arm]["post"].astype(np.float64)
        predicted = delta[arm]
        energy = float(np.sum(actual*actual))
        x = samples["dense"]["post"].astype(np.float64)
        ideal0 = x-x.mean(axis=1,keepdims=True)
        ideal0 /= np.sqrt(np.mean(ideal0**2,axis=1,keepdims=True)+1e-6)
        changed = x-predicted
        changed -= changed.mean(axis=1,keepdims=True)
        changed /= np.sqrt(np.mean(changed**2,axis=1,keepdims=True)+1e-6)
        actual_norm_delta = samples[arm]["nextnorm"].astype(np.float64)-samples["dense"]["nextnorm"].astype(np.float64)
        actual_norm_energy = float(np.sum(actual_norm_delta**2))
        native_y_delta = (samples["dense"]["ffn_output"].astype(np.float64)-
                          samples[arm]["ffn_output"].astype(np.float64))*a["gate"]
        product_delta = products["dense"]-products[arm]
        errors = np.stack([native_y_delta-predicted,product_delta-native_y_delta,actual-product_delta])
        gram = np.einsum("ird,jrd->ij",errors,errors)
        full_error = float(np.sum((actual-predicted)**2))
        if abs(gram.sum()-full_error)>1e-10*max(1.,energy):
            raise ValueError("Numerical-boundary error decomposition failed")
        arm_state = samples[arm]["post"].astype(np.float64)
        arm_centered = arm_state-arm_state.mean(axis=1,keepdims=True)
        arm_normalized = arm_centered/np.sqrt(np.mean(arm_centered**2,axis=1,keepdims=True)+1e-6)
        ideal_from_actual_state = arm_normalized-ideal0
        result["arms"][arm] = {
            "decoded_group_energy": float(np.sum(predicted**2)), "actual_writeback_SSE": energy,
            "decoded_direction_vs_native_writeback_relative_SSE": float(np.sum((actual-predicted)**2))/energy,
            "actual_nextnorm_SSE": actual_norm_energy,
            "ideal_LN_prediction_SSE": float(np.sum((changed-ideal0)**2)),
            "ideal_LN_prediction_vs_native_response_relative_SSE": float(np.sum((changed-ideal0-actual_norm_delta)**2))/actual_norm_energy,
            "ideal_LN_actual_writeback_response_SSE": float(np.sum(ideal_from_actual_state**2)),
            "ideal_LN_actual_writeback_vs_native_relative_SSE": float(np.sum((ideal_from_actual_state-actual_norm_delta)**2))/actual_norm_energy,
            "ideal_LN_actual_writeback_then_BF16_vs_native_relative_SSE": float(np.sum(
                (cast_norms[arm]-cast_norms["dense"]-actual_norm_delta)**2))/actual_norm_energy,
            "numerical_boundary_error_Gram": gram.tolist(),
            "numerical_boundary_order": ["native_FFN2_output_vs_decoded_panel","BF16_gate_multiply","BF16_residual_add"],
            "numerical_boundary_component_energy_over_actual_writeback_SSE": (np.diag(gram)/energy).tolist(),
            "numerical_boundary_signed_share_of_total_prediction_error": (gram.sum(axis=1)/full_error).tolist()}
    return result


def analyze(root, observation_root, indices, include_videos=False):
    manifest = json.loads((ROOT/"research/20261025/E0_NATIVE_OBSERVATIONS.json").read_text())
    result = {"status": "COMPLETE_FIXED_E0_NATIVE_FINITE_DIAGNOSTIC" if indices==manifest["source_indices"] else "PARTIAL_NATIVE_DIAGNOSTIC",
              "indices": indices, "source_clusters": len(indices), "tool_sha256": sha256(Path(__file__)),
              "scope": "Fixed groups0/53, step0/block0, unchanged full dense GEMMs. Native residual and nextnorm, then four free-running steps. Exposed sources, no speed or quality acceptance. Not amplitude-matched radial interventions.",
              "quality_or_speed_acceptance": False, "sources": []}
    for index in indices:
        source = next(s for s in manifest["sources"] if s["index"]==index)
        folder = root/f"source{index:02d}_attempt1"
        run_path = folder/"RUN.json"
        run = json.loads(run_path.read_text())
        if run["status"] != "PASS_FIXED_E0_FINITE_NATIVE_DIAGNOSTIC" or run["generations_started"]!=5 or list(run["phases"])!=ARMS:
            raise ValueError("Incomplete native fixed scope")
        if run["assigned_GPU_resources"]["violations"]:
            raise ValueError("GPU isolation violation")
        protocol = ROOT/("config/e0_finite_group_diagnostic_20261025.json" if index in [0,4] else "config/e0_finite_group_replication_20261025.json")
        if sha256(protocol)!=run["protocol_sha256"]:
            raise ValueError("Frozen execution protocol differs")
        p = json.loads(protocol.read_text())
        if any(sha256(folder/"EXECUTED_SOURCES"/name)!=expected for name,expected in p["driver_sources_sha256"].items()):
            raise ValueError("Executed source snapshot differs")
        data = {}
        for arm,phase in run["phases"].items():
            if phase["native_counts"] != source["native_counts_each_phase"] or set(phase["observer_calls"].values())!={1}:
                raise ValueError("Changed native invocation counts")
            if phase["prefix"]["pre_FFN_writeback_sha256"]!=run["phases"]["dense"]["prefix"]["pre_FFN_writeback_sha256"]:
                raise ValueError("Changed prefix")
            if sha256(folder/f"{arm}.mp4")!=phase["video_sha256"]:
                raise ValueError("Native video changed")
            data[arm] = []
            for entry in phase["trajectory"]:
                path = folder/entry["path"]
                if sha256(path)!=entry["sha256"]:
                    raise ValueError("Native trajectory payload changed")
                data[arm].append(load_arrays(path))
        for step in range(4):
            if any(not np.array_equal(data["dense"][step][k],data["dense_repeat"][step][k]) for k in ["input","flow","post"]):
                raise ValueError("Repeated dense trajectory control failed")
        for arm in ARMS:
            if not np.array_equal(data[arm][0]["input"],data["dense"][0]["input"]):
                raise ValueError("First-step prefixes differ")
        local = {arm: run["phases"][arm].get("native_full_rows_SSE",{}) for arm in ["A","B","AB"]}
        row = {"index":index,"source_group":source["source_group"],"run_sha256":sha256(run_path),
               "protocol_sha256":run["protocol_sha256"],"assets":run["assets"],
               "HF_relative":f"ffnds-derived/e0_finite_groups_20261025/source{index:02d}_attempt1",
               "generations":5,"native_counts_each":source["native_counts_each_phase"],
               "sampled_GPU_isolation_pass":True,"historical_dense_video_SHA_match":run["phases"]["dense"]["video_sha256"]==source["video_sha256"],
               "repeated_dense_bitwise":True,"native_full_rows_SSE":local,
               "native_local_interaction":run["native_interaction"],"steps":[],
               "sample_direction_check":sample_directions(source,folder,observation_root)}
        if include_videos:
            row["decoded_video_difference"] = decoded_video_distances(folder)
        for step in range(4):
            values = {arm: {k:squared_response(data[arm][step][k],data["dense"][step][k])
                            for k in ["flow","post"]} for arm in ["A","B","AB"]}
            signed = {k:four_arm(*(data[arm][step][k] for arm in ["dense","A","B","AB"])) for k in ["flow","post"]}
            row["steps"].append({"step":step,"responses":values,"interaction":signed})
        result["sources"].append(row)
    result["total_generations"] = sum(s["generations"] for s in result["sources"])
    result["rankings_A_vs_B"] = [{"index":s["index"],
        "local_post_A_over_B":s["native_full_rows_SSE"]["A"]["post"]/s["native_full_rows_SSE"]["B"]["post"],
        "nextnorm_A_over_B":s["native_full_rows_SSE"]["A"]["nextnorm"]/s["native_full_rows_SSE"]["B"]["nextnorm"],
        "flow_A_over_B_by_step":[t["responses"]["A"]["flow"]["SSE"]/t["responses"]["B"]["flow"]["SSE"] for t in s["steps"]],
        "final_latent_A_over_B":s["steps"][-1]["responses"]["A"]["post"]["SSE"]/s["steps"][-1]["responses"]["B"]["post"]["SSE"]}
        for s in result["sources"]]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root",type=Path,required=True)
    p.add_argument("--observation-root",type=Path,required=True)
    p.add_argument("--indices",type=int,nargs="+",default=[0,4,8,12,16,20])
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--include-videos",action="store_true")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Preserve the first result")
    write_json(a.out,analyze(a.raw_root,a.observation_root,a.indices,a.include_videos))


if __name__=="__main__":
    main()
