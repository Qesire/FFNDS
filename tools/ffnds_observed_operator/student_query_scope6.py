"""Replay the independent BF16 Dense teacher on the 18 scope-six hit panels."""

import argparse
import csv
from pathlib import Path

from safetensors import safe_open
import torch

from .local_data import (
    ORIGINAL_BINDINGS, bound_json, checked, native_runtime, require, sha_file, tensor_contract, tensor_sha,
)
from .local_worker import allocation, event, write_json
from .quality_scope6 import ARMS, LAYERS, SOURCE, proof
from .student_query import GLOBAL_P64_SCOPE_SHA, QUERY_COUNT, WIDTH, load_observer, measure, read, row_ids

PLACEMENTS = 18


def load_operand(cell, arm, site, probe_ids, query_ids, hashes):
    path = checked(cell["asset"])
    data = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
    p64 = arm == "Global-P64-6"
    schema = "FFNDS_OBSERVED_SCOPE6_QUERY_V1" if p64 else "FFNDS_OBSERVED_FULL80_QUERY_V1"
    step = 1 if site.startswith("high") else 3
    require((data["schema"], data["source"], data["arm"], data["site"], data["step"]) ==
            (schema, SOURCE, arm, site, step)
            and data["identity"]["ffn_tensor_sha256"] == hashes
            and data["probe_calls"] == int(arm != ARMS[0]),
            f"{path}: wrong saved producer, site, or FFN weights")
    if p64:
        require(data["probe_kind"] == "native_w1_only"
                and data["identity"]["global_p64_scope"]["sha256"] == GLOBAL_P64_SCOPE_SHA,
                f"{path}: wrong historical Global-P64 producer")
    else:
        require(data["adaptation"] is (arm == ARMS[1]), f"{path}: wrong readout adaptation")
    for name, ids in (("probe_ids", probe_ids), ("query_ids", query_ids)):
        value = data[name]
        require(tensor_contract(value, (QUERY_COUNT,), torch.int64) and value.tolist() == ids,
                f"{path}: wrong {name} raster positions")
    for name, value, shape in (("u1", data["u1"], (QUERY_COUNT, WIDTH)),
                               ("y_pred", data["y_pred"], (QUERY_COUNT, WIDTH)),
                               ("gate", data["gate"], (WIDTH,)),
                               ("y0", data["anchor"]["y0"], (QUERY_COUNT, WIDTH))):
        require(tensor_contract(value, shape, torch.bfloat16) and bool(torch.isfinite(value).all()),
                f"{path}: invalid BF16 {name}")
    require(tensor_sha(data["gate"]) == cell["gate_sha256"], f"{path}: wrong current gate bytes")
    return data


def collect(scope, archive):
    stages = scope["producer_stages"]
    require(set(stages) == set(ARMS) and set(stages.values()) == set(scope["generation_exits"])
            and len(set(stages.values())) == 3, "Expected exactly three distinct completed scope-six generation stages")
    candidates, common, readout_release = {}, None, None
    for arm in ARMS:
        path = Path(stages[arm])
        exit_ref = scope["generation_exits"][str(path)]
        require(path.is_absolute() and Path(exit_ref["path"]) == path / "EXIT.json", "Wrong registered EXIT locator")
        gen, binding, video, cells, job, _ = proof(path, arm, exit_ref["sha256"])
        fixed = binding["global_p64_scope"]
        require(fixed["sha256"] == GLOBAL_P64_SCOPE_SHA and gen["bindings"] == scope["bindings"]
                and gen["experiment_spec"]["sha256"] == scope["experiment_spec"]["sha256"]
                and gen["schedule"] == archive["schedule"]
                and {k: (v["path"], v["sha256"]) for k, v in binding["checkpoints"].items()} ==
                {k: (v["path"], v["sha256"]) for k, v in archive["checkpoints"].items()},
                f"{arm}: wrong scope-six generation dependencies")
        canonical = bound_json(fixed)["fixed64_token_ids"]
        probe, query = row_ids(canonical)
        require((binding["canonical_global_p64_ids"], binding["fixed_probe_ids"], binding["query_ids"]) ==
                (canonical, probe, query), f"{arm}: frozen probe/query IDs changed")
        checked(gen["experiment_spec"])
        base = {k: gen[k] for k in ("bindings", "parameters", "dense_initial_noise", "schedule",
                                   "strict_effective_config", "native_transformer_sha256", "reference_sha256")}
        if common is None:
            common = base, fixed, query
        require(common == (base, fixed, query), f"{arm}: cross-arm bases or query IDs differ")
        if arm != "Global-P64-6":
            release = gen["full80_parameters"]
            if readout_release is None:
                readout_release = release
            require(release == readout_release, "Readout scope-six arms used different CAL releases")
        used_probe = canonical if arm == "Global-P64-6" else probe
        candidates[arm] = {"queries": cells, "probe": used_probe, "query": query, "job_id": job,
                           "exit": exit_ref, "video_sha256": video["video"]["mp4_sha256"]}
    require(len({r["job_id"] for r in candidates.values()}) == 3, "Scope-six generation jobs alias")
    return candidates


@torch.inference_mode()
def run(scope_path):
    scope = read(scope_path)
    require(scope["method"] == "OBSERVED_SECANT_SCOPE6_STUDENT_QUERY" and Path(scope["stage"]) == Path.cwd()
            and scope["max_d4"] == scope["max_nfe"] == 0 and scope["max_query_placements"] == PLACEMENTS
            and scope["max_query_token_ffn"] == PLACEMENTS * QUERY_COUNT
            and scope["bindings"]["sha256"] == ORIGINAL_BINDINGS, "Wrong bounded scope-six teacher scope")
    spec = bound_json(scope["experiment_spec"])
    require(spec["student_query"]["count"] == QUERY_COUNT
            and (spec["budget"]["offline_student_query_token_ffn_all_max"]
                 - spec["budget"]["offline_student_query_token_ffn_main_max"]) == PLACEMENTS * QUERY_COUNT,
            "Amended teacher budget differs")
    archive = bound_json(scope["bindings"])
    require(archive["schema"] == "FFNDS_STATE_OPERATOR_BINDINGS_V1" and archive["status"] == "PASS",
            "Wrong frozen checkpoint archive")
    device, runtime = allocation(scope)
    out = Path("output")
    out.mkdir()
    event(out, "VERIFY_SCOPE6_GENERATIONS", required_videos=3)
    candidates = collect(scope, archive)
    factory, _, _ = native_runtime(scope, archive)
    require(torch.get_float32_matmul_precision() == "highest" and not torch.backends.cuda.matmul.allow_tf32
            and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, "Native import changed precision")
    results, writer, calls = [], None, 0
    csv_path = out / "STUDENT_QUERY.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        for expert, step in (("high", 1), ("low", 3)):
            checkpoint = archive["checkpoints"][expert]
            with safe_open(checked(checkpoint), framework="pt", device="cpu") as container:
                for layer in LAYERS:
                    site = f"{expert}{layer:02d}"
                    observer, hashes = load_observer(container, layer, device, factory)
                    for arm, candidate in candidates.items():
                        cell = candidate["queries"][site]
                        data = load_operand(cell, arm, site, candidate["probe"], candidate["query"], hashes)
                        require(calls < PLACEMENTS, "Scope-six teacher budget exceeded")
                        calls += 1
                        event(out, "TEACHER_CALL_START", arm=arm, site=site, calls=calls,
                              token_ffn=calls * QUERY_COUNT)
                        teacher = observer(data["u1"].to(device))
                        metrics = measure(data["y_pred"].to(device), teacher, data["anchor"]["y0"].to(device),
                                          data["gate"].to(device))
                        row = {"source": SOURCE, "arm": arm, "expert": expert, "layer": layer, "step": step,
                               "query_rows": QUERY_COUNT, "payload_sha256": cell["asset"]["sha256"],
                               "query_ids_sha256": tensor_sha(data["query_ids"]),
                               "probe_ids_sha256": tensor_sha(data["probe_ids"]),
                               "student_input_sha256": tensor_sha(data["u1"]),
                               "student_prediction_sha256": tensor_sha(data["y_pred"]),
                               "teacher_output_sha256": tensor_sha(teacher), "gate_sha256": cell["gate_sha256"],
                               "checkpoint_sha256": checkpoint["sha256"], "generation_job_id": candidate["job_id"], **metrics}
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        stream.flush()
                        results.append(row)
    require(len(results) == calls == PLACEMENTS, "Incomplete scope-six teacher placements")
    aggregate = {}
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        error, anchor, count = (sum(r[k] for r in rows) for k in ("error_energy", "teacher_increment_energy", "value_count"))
        aggregate[arm] = {"placements": len(rows), "gated_mse": error / count,
                          "gated_anchor_mse": anchor / count, "relative_risk_to_own_anchor": error / anchor if anchor else None}
    write_json(out / "STUDENT_QUERY_SUMMARY.json", {
        "schema": "FFNDS_OBSERVED_SCOPE6_STUDENT_QUERY_V1", "status": "COMPLETE", "scope_sha256": sha_file(scope_path),
        "source": SOURCE, "teacher": "Independent native BF16 W1/GELU/W2 on each saved student U1[64]",
        "teacher_batch_shape": [QUERY_COUNT, WIDTH], "placements": len(results), "teacher_calls": calls,
        "teacher_token_ffn": calls * QUERY_COUNT, "nfe": 0, "d4": 0, "runtime": runtime,
        "native_sources": scope["native_sources"], "results": aggregate,
        "readout_online_minus_static_gated_mse_on_own_states": aggregate[ARMS[1]]["gated_mse"] - aggregate[ARMS[0]]["gated_mse"],
        "generation_inputs": {arm: {k: r[k] for k in ("exit", "job_id", "video_sha256", "probe", "query")}
                              for arm, r in candidates.items()},
        "per_site_csv": {"path": str(csv_path.absolute()), "sha256": sha_file(csv_path), "bytes": csv_path.stat().st_size},
    })
    event(out, "TEACHER_COMPLETE", calls=calls, token_ffn=calls * QUERY_COUNT, nfe=0, d4=0)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    raise SystemExit(run(parser.parse_args().scope))
