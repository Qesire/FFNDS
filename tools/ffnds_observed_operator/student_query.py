"""Replay independent Wan Dense BF16 teachers on saved full80 student queries."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch
from torch.nn.functional import gelu

from .local_data import (
    ORIGINAL_BINDINGS, SHAPES, SUFFIXES, bound_json, checked, native_linear, native_runtime,
    require, sha_file, tensor_contract, tensor_sha,
)
from .local_worker import allocation, event, write_json


ARMS = ("readout-static-80", "readout-online-80", "modes-static-80", "modes-online-80")
DEV3 = ("vbi_0392", "vbi_1013", "vbi_0800")
EXPERT_STEPS = {"high": 1, "low": 3}
AMENDMENT = "58f68ab460d47f0f803fd3f2f1b6ec03b651ec21"
GLOBAL_P64_SCOPE_SHA = "abcbf252c09aafcf4df89d8bcd90addd46f0d8bedb75c8bfbad9ec0628595e0c"
PLACEMENTS, QUERY_COUNT, WIDTH = 960, 64, 5120


def read(path):
    return json.loads(path.read_bytes())


def row_ids(canonical):
    available = np.arange(31668, dtype=np.int64)
    require(len(canonical) == len(set(canonical)) == QUERY_COUNT
            and all(isinstance(i, int) and 0 <= i < len(available) for i in canonical), "Canonical P64 IDs differ")
    probe = np.random.default_rng(42).permutation(available)[:QUERY_COUNT]
    eligible = np.setdiff1d(available, np.union1d(probe, canonical))
    query = np.random.default_rng(42).permutation(eligible)[:QUERY_COUNT]
    return probe.tolist(), query.tolist()


def verified_stage(stage, exit_ref, arm, archive):
    require(stage.is_absolute() and Path(exit_ref["path"]).is_absolute()
            and Path(exit_ref["path"]).samefile(stage / "EXIT.json"), f"{stage}: wrong EXIT locator")
    exit_record = bound_json(exit_ref)
    scope_path = stage / "scope.json"
    scope_sha = sha_file(scope_path)
    scope = read(scope_path)
    sources = tuple(scope["sources"])
    require(scope["base_commit"] == AMENDMENT and scope["method"] == "WAN_FREE_GENERATION"
            and scope["arm"] == arm and scope["seed"] == 42 and scope["retries"] == 0
            and scope["force_dense"] is False and (sources == DEV3 or len(sources) == 1 and sources[0] in DEV3)
            and (scope["max_d4"], scope["max_nfe"]) == (len(sources), 4 * len(sources))
            and scope["bindings"]["sha256"] == ORIGINAL_BINDINGS, f"{stage}: wrong full80 generation scope")
    run = read(stage / "RUN.json")
    attempt = read(stage / "ATTEMPT_EXIT.json")
    review = read(stage / "REVIEW.json")
    active = read(stage / "ACTIVE_RESEARCH.json")[scope["scope_id"].lower()]
    review_sha = sha_file(stage / "REVIEW.json")
    job = str(run["slurm_job_id"])
    require(exit_record["exit_code"] == attempt["exit_code"] == 0
            and (run["scope_id"], exit_record["scope_id"]) == (scope["scope_id"], scope["scope_id"])
            and all(str(row["slurm_job_id"]) == job for row in (exit_record, attempt))
            and all(row["scope_sha256"] == scope_sha for row in (run, attempt, review, active))
            and active["runnable"] is True and active["review_sha256"] == run["review_sha256"] == review_sha
            and review["recheck_disposition"] == "APPROVE"
            and review["reviewer_agent_id"] and review["reviewer_agent_id"] != review["author_agent_id"],
            f"{stage}: missing valid reviewed EXIT0")
    outputs = exit_record["output_sha256"]
    def load(name):
        return bound_json({"path": str(stage / name), "sha256": outputs[name]})
    native = load("receipt/RUN.json")
    binding = load("output/BINDINGS.json")
    summary = load("output/SUMMARY.json")
    checkpoints = {expert: (ref["path"], ref["sha256"]) for expert, ref in archive["checkpoints"].items()}
    require(native["scope_sha256"] == binding["scope_sha256"] == scope_sha
            and str(native["allocation"]["job_id"]) == str(binding["slurm_job_id"]) == job
            and binding["schema"] == "FFNDS_OBSERVED_FULL80_BINDINGS_V1"
            and {expert: (ref["path"], ref["sha256"]) for expert, ref in binding["checkpoints"].items()} == checkpoints
            and binding["arm"] == arm
            and binding["sources"] == list(sources) and binding["parameters"] == scope["full80_parameters"]
            and binding["p0"] == scope["bindings"] and binding["strict_q1"] == scope["parameters"]
            and [r["source"] for r in summary["records"]] == list(sources)
            and (summary["status"], summary["arm"], summary["nfe"]) == ("COMPLETE", arm, 4 * len(sources)),
            f"{stage}: native generation proof differs")
    canonical_ref = scope["canonical_gp64_scope"]
    require(canonical_ref["sha256"] == GLOBAL_P64_SCOPE_SHA, f"{stage}: canonical P64 scope differs")
    canonical = bound_json(canonical_ref)["fixed64_token_ids"]
    probe, query = row_ids(canonical)
    require(canonical == scope["fixed64_token_ids"] == binding["canonical_global_p64_ids"]
            and probe == binding["fixed_probe_ids"] and query == binding["query_ids"], f"{stage}: fixed IDs changed")
    contract = {k: scope[k] for k in ("bindings", "parameters",
                                     "strict_effective_config", "reference_sha256", "schedule", "dense_initial_noise",
                                     "native_transformer_sha256", "canonical_gp64_scope")}
    contract["experiment_spec_sha256"] = scope["experiment_spec"]["sha256"]
    checked(scope["experiment_spec"])
    return job, load, summary, contract, probe, query, outputs, scope["full80_parameters"]


def collect(scope, archive):
    stages = scope["producer_stages"]
    require(set(stages) == set(ARMS), "Expected all four executed full80 arms")
    require(all(isinstance(entry, str) or isinstance(entry, dict) and set(entry) == set(DEV3)
                for entry in stages.values()), "Full80 stage map must cover DEV3")
    locators = {entry if isinstance(entry, str) else path
                for entry in stages.values() for path in ([entry] if isinstance(entry, str) else entry.values())}
    require(set(scope["generation_exits"]) == locators, "Unbound or unexpected generation EXIT")
    cells, by_stage, family_releases, manifests, shared, initial_noise = {}, {}, {}, {}, None, None
    site_keys = {f"{expert}{layer:02d}" for expert in EXPERT_STEPS for layer in range(40)}
    video_proofs = []
    for arm in ARMS:
        for source in DEV3:
            loc = stages[arm] if isinstance(stages[arm], str) else stages[arm][source]
            stage = Path(loc)
            if (loc, arm) not in by_stage:
                by_stage[loc, arm] = verified_stage(stage, scope["generation_exits"][loc], arm, archive)
            job, load, summary, contract, probe, query, outputs, release = by_stage[loc, arm]
            family = arm.split("-", 1)[0]
            require(family_releases.setdefault(family, release) == release,
                    f"{stage}: {family} static and online must share one CAL release")
            if family not in manifests:
                manifest = bound_json(release)
                family_key = "observed_readout" if family == "readout" else "native_modes"
                require(manifest["schema"] == "FFNDS_OBSERVED_SECANT_FULL80_PARAMETERS_V1"
                        and manifest["status"] == "PASS" and set(manifest["sites"]) == site_keys
                        and manifest["families"][family_key]["ready"] is True
                        and manifest["bindings"] == contract["bindings"]
                        and manifest["strict_q1"] == contract["parameters"]
                        and manifest["spec_sha256"] == scope["experiment_spec"]["sha256"],
                        f"{stage}: CAL release does not match the generation base")
                manifests[family] = manifest
            if shared is None:
                shared = contract
                initial_noise = bound_json(contract["dense_initial_noise"])
            require(contract == shared and contract["experiment_spec_sha256"] == scope["experiment_spec"]["sha256"],
                    f"{stage}: mixed generation recipes or CAL identity")
            src_json = f"output/{source}.json"
            src_mp4 = f"output/{source}.mp4"
            receipt = load(src_json)
            video = receipt["video"]
            require((receipt["source"], receipt["arm"], receipt["seed"]) == (source, arm, 42)
                    and receipt["schedule"] == archive["schedule"]
                    and receipt["initial_noise"] == initial_noise[source]
                    and receipt["dispatch"] == [{"step": i, "expert": i // 2} for i in range(4)]
                    and receipt["ffn_calls_by_step"] == [40] * 4
                    and tuple(video[k] for k in ("frames", "fps", "height", "width")) == (81, 16, 464, 832)
                    and outputs[src_mp4] == video["mp4_sha256"] == sha_file(stage / src_mp4)
                    and next(r["receipt_sha256"] for r in summary["records"] if r["source"] == source) == outputs[src_json],
                    f"{stage}: mismatched completed video receipt for {source}")
            rows, gates = receipt["sites"], receipt["gate_witness"]
            require(len(rows) == len(gates) == 160, f"{stage}: incomplete full80 trajectory")
            for i, (row, gate) in enumerate(zip(rows, gates, strict=True)):
                step, layer = divmod(i, 40)
                expert = "high" if step < 2 else "low"
                hit = step == EXPERT_STEPS[expert]
                require((row["expert"], row["step"], row["layer"], row["operator"], row["kind"], row["probe_calls"]) ==
                        (expert, step, layer, arm, "hit" if hit else "anchor", int(hit and "-online-" in arm))
                        and gate["full_write_bitwise_equal"] is True and gate["gate_dtype"] == "torch.bfloat16"
                        and all(gate[k] == v for k, v in row.items()), f"{stage}: incorrect full80 site or gate at {step}/{layer}")
                if hit:
                    path = f"output/{source}_query/{expert}{layer:02d}.pt"
                    ref = row["query"]
                    expected = stage / path
                    require(Path(ref["path"]).is_absolute() and Path(ref["path"]).samefile(expected)
                            and ref["sha256"] == outputs[path]
                            and expected.is_file() and expected.stat().st_size == ref["bytes"],
                            f"{stage}: unbound student query at {step}/{layer}")
                    cells[expert, layer, source, arm] = {
                        "asset": ref, "gate_sha256": gate["gate_sha256"], "probe": probe, "query": query,
                        "job_id": job, "video_sha256": video["mp4_sha256"], "video_receipt_sha256": outputs[src_json],
                        "family_parameters": manifests[family]["sites"][f"{expert}{layer:02d}"],
                    }
            video_proofs.append({"source": source, "arm": arm, "video_sha256": video["mp4_sha256"],
                                 "generation_job_id": job, "generation_exit": scope["generation_exits"][loc],
                                 "cal_release": release})
    require(len({(row["generation_job_id"], row["source"]) for row in video_proofs}) == 12,
            "One generated source/job is aliased across multiple arms")
    require(len(cells) == PLACEMENTS, "Unexpected student query placement count")
    return cells, video_proofs


def load_query(cell, key):
    expert, layer, source, arm = key
    asset = checked(cell["asset"])
    data = torch.load(asset, mmap=True, map_location="cpu", weights_only=True)
    require((data["schema"], data["source"], data["arm"], data["site"], data["step"]) ==
            ("FFNDS_OBSERVED_FULL80_QUERY_V1", source, arm, f"{expert}{layer:02d}", EXPERT_STEPS[expert])
            and data["probe_calls"] == int("-online-" in arm) and data["adaptation"] is ("-online-" in arm)
            and data["identity"]["family_parameters"] == cell["family_parameters"],
            f"{asset}: invalid student payload identity")
    for name, ids in (("probe", cell["probe"]), ("query", cell["query"])):
        operand = data[f"{name}_ids"]
        require(tensor_contract(operand, (QUERY_COUNT,), torch.int64) and operand.tolist() == ids,
                f"{asset}: saved {name} IDs differ")
    for name, operand, shape in (("u1", data["u1"], (QUERY_COUNT, WIDTH)),
                                 ("y_pred", data["y_pred"], (QUERY_COUNT, WIDTH)),
                                 ("gate", data["gate"], (WIDTH,)),
                                 ("y0", data["anchor"]["y0"], (QUERY_COUNT, WIDTH))):
        require(tensor_contract(operand, shape, torch.bfloat16) and bool(torch.isfinite(operand).all()),
                f"{asset}: {name} is not a finite contiguous BF16 tensor")
    require(tensor_sha(data["gate"]) == cell["gate_sha256"], f"{asset}: current gate differs from native receipt")
    return data


def load_observer(checkpoint, layer, device, factory):
    prefix = f"blocks.{layer}.ffn"
    operands, hashes = [], {}
    for suffix, shape in zip(SUFFIXES, SHAPES, strict=True):
        name = f"{prefix}.{suffix}"
        value = checkpoint.get_tensor(name)
        require(tensor_contract(value, shape, torch.bfloat16) and bool(torch.isfinite(value).all()),
                f"{name}: wrong BF16 checkpoint operand")
        hashes[name] = tensor_sha(value)
        operands.append(value.to(device))
    w1, b1, w2, b2 = operands
    first = native_linear(w1, b1, f"{prefix}.0", factory)
    second = native_linear(w2, b2, f"{prefix}.2", factory)

    def observe(u):
        require(tensor_contract(u, (QUERY_COUNT, WIDTH), torch.bfloat16) and u.device == device,
                "Teacher expects only the saved BF16 student columns and 64 rows")
        y = second(gelu(first(u), approximate="tanh"))
        require(tensor_contract(y, (QUERY_COUNT, WIDTH), torch.bfloat16) and bool(torch.isfinite(y).all()),
                "Independent native BF16 teacher returned invalid output")
        return y

    return observe, hashes


def measure(pred, teacher, anchor, gate):
    gate = gate.double()
    error = (pred.double() - teacher.double()) * gate
    increment = (teacher.double() - anchor.double()) * gate
    error_energy, increment_energy = error.square().sum().item(), increment.square().sum().item()
    gated_mse = error_energy / teacher.numel()
    anchor_mse = increment_energy / teacher.numel()
    risk = error_energy / increment_energy if increment_energy > 0 else None
    require(all(math.isfinite(x) for x in (error_energy, increment_energy, gated_mse, anchor_mse))
            and (risk is None or math.isfinite(risk)), "Nonfinite gated query metric")
    return {
        "gated_mse": gated_mse, "gated_anchor_mse": anchor_mse,
        "relative_risk_to_own_anchor": risk, "error_energy": error_energy,
        "teacher_increment_energy": increment_energy, "value_count": teacher.numel(),
    }


def summarize(rows):
    source_arms = []
    for source in DEV3:
        for arm in ARMS:
            group = [r for r in rows if (r["source"], r["arm"]) == (source, arm)]
            require(len(group) == 80, "Missing query teacher placements")
            error = sum(r["error_energy"] for r in group)
            anchor = sum(r["teacher_increment_energy"] for r in group)
            elements = sum(r["value_count"] for r in group)
            source_arms.append({"source": source, "arm": arm, "placements": len(group),
                                "gated_mse": error / elements, "gated_anchor_mse": anchor / elements,
                                "relative_risk_to_own_anchor": error / anchor if anchor > 0 else None})
    paired = []
    for source in DEV3:
        for family in ("readout", "modes"):
            arms = {r["arm"]: r for r in source_arms if r["source"] == source}
            online, static = (arms[f"{family}-{condition}-80"] for condition in ("online", "static"))
            paired.append({"source": source, "family": family,
                           "online_minus_static_gated_mse_on_own_states": online["gated_mse"] - static["gated_mse"]})
    return source_arms, paired


@torch.inference_mode()
def run(scope_path):
    scope = read(scope_path)
    require(scope["method"] == "OBSERVED_SECANT_FULL80_STUDENT_QUERY" and Path(scope["stage"]) == Path.cwd()
            and scope["max_d4"] == scope["max_nfe"] == 0
            and scope["max_query_placements"] == PLACEMENTS and scope["max_query_token_ffn"] == PLACEMENTS * QUERY_COUNT
            and scope["bindings"]["sha256"] == ORIGINAL_BINDINGS, "Invalid bounded teacher scope")
    spec = bound_json(scope["experiment_spec"])
    require(spec["cohorts"]["DEV3"] == list(DEV3)
            and spec["student_query"]["count"] == QUERY_COUNT
            and spec["budget"]["offline_student_query_token_ffn_main_max"] == PLACEMENTS * QUERY_COUNT,
            "Frozen student-query design differs")
    archive = bound_json(scope["bindings"])
    require(archive["schema"] == "FFNDS_STATE_OPERATOR_BINDINGS_V1" and archive["status"] == "PASS",
            "No frozen P0 source/checkpoint archive")
    device, allocation_record = allocation(scope)
    out = Path("output")
    out.mkdir()
    event(out, "BIND_COMPLETED_GENERATIONS", required_videos=12)
    cells, videos = collect(scope, archive)
    factory, _, _ = native_runtime(scope, archive)
    require(torch.get_float32_matmul_precision() == "highest" and not torch.backends.cuda.matmul.allow_tf32
            and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, "Native import changed precision")
    rows, writer, attempt_count = [], None, 0
    csv_path = out / "STUDENT_QUERY.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        for expert, step in EXPERT_STEPS.items():
            cp = archive["checkpoints"][expert]
            checkpoint = checked(cp)
            with safe_open(checkpoint, framework="pt", device="cpu") as container:
                for layer in range(40):
                    observe, hashes = load_observer(container, layer, device, factory)
                    for source in DEV3:
                        for arm in ARMS:
                            key = expert, layer, source, arm
                            cell = cells[key]
                            student = load_query(cell, key)
                            require(student["identity"]["ffn_tensor_sha256"] == hashes,
                                    f"{key}: teacher checkpoint differs from the live student FFN identity")
                            u = student["u1"].to(device)
                            require(attempt_count < PLACEMENTS, "Teacher call exceeds authorized budget")
                            attempt_count += 1
                            try:
                                teacher = observe(u)
                                values = measure(student["y_pred"].to(device), teacher,
                                                 student["anchor"]["y0"].to(device), student["gate"].to(device))
                            except (ValueError, RuntimeError):
                                event(out, "TEACHER_CALL_FAILED", expert=expert, layer=layer, source=source, arm=arm,
                                      attempts=attempt_count, completed=len(rows), token_ffn=attempt_count * QUERY_COUNT)
                                raise
                            row = {"source": source, "arm": arm, "expert": expert, "layer": layer, "step": step,
                                   "query_rows": QUERY_COUNT, "query_ids_sha256": tensor_sha(student["query_ids"]),
                                   "probe_ids_sha256": tensor_sha(student["probe_ids"]),
                                   "query_payload_sha256": cell["asset"]["sha256"], "student_input_sha256": tensor_sha(u),
                                   "student_prediction_sha256": tensor_sha(student["y_pred"]),
                                   "teacher_output_sha256": tensor_sha(teacher), "gate_sha256": cell["gate_sha256"],
                                   "checkpoint_sha256": cp["sha256"], "generation_job_id": cell["job_id"],
                                   "video_sha256": cell["video_sha256"], "video_receipt_sha256": cell["video_receipt_sha256"],
                                   **values}
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append(row)
                    stream.flush()
                    event(out, "SITE_TEACHERS_COMPLETE", expert=expert, layer=layer,
                          attempts=attempt_count, completed=len(rows), token_ffn=attempt_count * QUERY_COUNT)
    require(attempt_count == len(rows) == PLACEMENTS, "Unexpected actual teacher consumption")
    source_arms, paired = summarize(rows)
    write_json(out / "STUDENT_QUERY_SUMMARY.json", {
        "schema": "FFNDS_OBSERVED_FULL80_STUDENT_QUERY_V1", "status": "COMPLETE", "scope_sha256": sha_file(scope_path),
        "nfe": 0, "d4": 0, "placements": len(rows), "teacher_calls": attempt_count,
        "teacher_token_ffn": attempt_count * QUERY_COUNT, "teacher_batch_shape": [QUERY_COUNT, WIDTH],
        "teacher": "Independent BF16 MMWeight.apply W1, tanh GELU and MMWeight.apply W2 on saved pre-W1 student inputs",
        "metric": "Current-gate weighted pre-gate error on each arm's own student trajectory. Relative risk uses own Dense anchor increment.",
        "runtime": allocation_record, "native_sources": scope["native_sources"], "checkpoints": archive["checkpoints"],
        "videos": videos, "query_ids": next(iter(cells.values()))["query"], "probe_ids": next(iter(cells.values()))["probe"],
        "per_site_csv": {"path": str(csv_path.absolute()), "sha256": sha_file(csv_path), "bytes": csv_path.stat().st_size},
        "source_arm_results": source_arms, "paired_source_deltas": paired,
    })
    event(out, "TEACHER_COMPLETE", placements=len(rows), teacher_token_ffn=attempt_count * QUERY_COUNT, nfe=0, d4=0)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    raise SystemExit(run(parser.parse_args().scope))
