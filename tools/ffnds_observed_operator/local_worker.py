"""Run the independently reviewed six-site observed-secant experiment on archived operands."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from shutil import disk_usage
import subprocess
import time

import numpy as np
import torch

from . import mode_local, observed_local
from .local_data import SITES, archive_inventory, bound_json, checked, fill_probes, load_pair, load_site, native_runtime, require, sha_file
from .local_metrics import PanelDirections, common_rows, select_families


def write_json(path, data):
    pending = path.with_suffix(path.suffix + ".partial")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.rename(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, tuple, dict)) else value
                             for key, value in row.items()})


def event(output, phase, **details):
    record = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "phase": phase, **details}
    with (output / "PROGRESS.jsonl").open("a", encoding="utf-8") as stream:
        print(json.dumps(record, allow_nan=False), file=stream)
    print(json.dumps(record, allow_nan=False), flush=True)


def allocation(scope):
    resource = scope["resources"]
    require(os.environ["SLURM_JOB_PARTITION"] == resource["partition"]
            and int(os.environ["SLURM_CPUS_PER_TASK"]) == resource["cpus"], "Slurm CPU or partition changed")
    job_id = os.environ["SLURM_JOB_ID"]
    record = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True).strip()
    fields = record.split()
    for key, value in {"Account": resource["account"], "QOS": resource["qos"], "JobId": job_id,
                       "MinMemoryNode": f"{resource['memory_gib']}G", "Requeue": 0, "NumCPUs": resource["cpus"],
                       "TimeLimit": resource["time"]}.items():
        require(f"{key}={value}" in fields, f"Slurm allocation differs at {key}")
    free = disk_usage(Path.cwd()).free
    require(free >= resource["disk_floor_gib"] * 1024**3, "Output archive lacks frozen free space")
    require(torch.cuda.device_count() == resource["gpus"] == 1, "Unexpected CUDA allocation")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    require(props.name == "NVIDIA A100-SXM4-80GB" and (props.major, props.minor) == (8, 0)
            and props.total_memory >= 78 * 1024**3, "Unexpected A100 hardware")
    require(torch.__version__ == scope["torch_version"] and torch.version.cuda == scope["torch_cuda"]
            and np.__version__ == scope["numpy_version"], "Frozen numeric packages changed")
    torch.cuda.set_per_process_memory_fraction(resource["gpu_memory_gib"] * 1024**3 / props.total_memory, device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            and torch.get_float32_matmul_precision() == "highest", "Runtime precision cannot be bound")
    return device, {"job_id": job_id, "scontrol": record, "device": props.name, "device_memory_bytes": props.total_memory,
                    "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__, "tf32": False,
                    "bf16_reduced_precision_reduction": True, "autocast": False, "output_free_bytes_at_entry": free}


def bind_fold_receipts(ids_by_source):
    from .local_data import folds_for, tensor_sha

    result = {}
    for source, ids in ids_by_source.items():
        result[source] = {"token_id_sha256": tensor_sha(ids), "folds": [
            {key: fold[key] for key in ("fold", "pool_ids", "query_ids", "probe_ids")}
            for fold in folds_for(source, ids, "cpu")]}
    return result


def observed_acceptance(result):
    acceptance = result["acceptance"]
    folds = acceptance.get("folds", [])
    valid = (acceptance["status"] == "VALID" and len(folds) == 44
             and all(row["static_gate_equal"] is True and row["fixed64_gate_equal"] is True for row in folds))
    return {**acceptance, "valid": valid}


def modes_acceptance(result):
    acceptance = result["acceptance"]
    valid = bool(acceptance["valid"] and acceptance["gate_witness_passed"] is True)
    return {**acceptance, "valid": valid, "status": result["status"]}


@torch.inference_mode()
def run(scope_path):
    scope = json.loads(scope_path.read_bytes())
    require(scope["method"] == "OBSERVED_SECANT_LOCAL6" and scope["max_nfe"] == scope["max_d4"] == 0,
            "Worker only implements the frozen LOCAL6 scope")
    device, runtime_info = allocation(scope)
    spec = bound_json(scope["experiment_spec"])
    require(spec["assets"]["six_site_pairs"] == 132 and spec["local_split"]["rows"] == 512
            and not spec["budget"]["unbudgeted_new_dense_capture"], "Local source contract changed")
    archive, bank, references = archive_inventory(scope, spec)
    runtime = native_runtime(scope, archive)
    require(torch.get_float32_matmul_precision() == "highest" and not torch.backends.cuda.matmul.allow_tf32
            and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, "Native import changed precision")
    output = Path("output")
    output.mkdir()
    (output / "site_parameters").mkdir()
    event(output, "ARCHIVE_HASH_CHECK", checkpoints=list(archive["checkpoints"]))
    checkpoints = {name: checked(ref) for name, ref in archive["checkpoints"].items()}

    query, structure = [], []
    acceptance, tunings, loaded_ids, site_identities = {}, {}, {}, {}
    parameters = {"schema": "FFNDS_OBSERVED_SECANT_LOCAL_CAL_V1", "source_role": "CAL11",
                  "sources": archive["cal_sources"], "base_matrix_policy": "L64_AT_ALL_SIX_BOUND_SITES", "sites": {}}
    sources_by_split = {split: archive[f"{split.lower()}_sources"] for split in ("CAL", "DEV")}
    for expert, layer in SITES:
        key = f"{expert}{layer:02d}"
        site_start = time.monotonic()
        event(output, "SITE_LOAD", site=key)
        site = load_site(expert, layer, archive, checkpoints[expert], device, runtime)
        site_identities[key] = site["identity"]
        panels = defaultdict(list)
        for split, names in sources_by_split.items():
            for source in names:
                pair = load_pair(references[expert, layer, source], archive, bank, loaded_ids, device)
                fill_probes(site, pair)
                panels[split].append(pair)
        direction = site["direction"] = PanelDirections(site)
        common, fixed_observers = common_rows(site, panels["CAL"] + panels["DEV"])
        event(output, "READOUT_CAL_DEV", site=key)
        readout = observed_local.process_site(site, panels["CAL"], panels["DEV"])
        event(output, "MODE_CAL_DEV", site=key, readout_status=readout["acceptance"]["status"])
        modes = mode_local.process_site(site, panels["CAL"], panels["DEV"])
        # Always retain the structurally independent panel-P64 control at the full cohort.
        for pairs in panels.values():
            for pair in pairs:
                if (pair["source"], 1) not in direction.cache:
                    direction._prepare(pair)
        for mode_row in modes["rows"]:
            mode_row.pop("direction", None)  # The unified direction table carries these rows.
        site_rows = common + readout["rows"] + modes["rows"]
        site_structure = readout["structure"] + direction.rows
        query.extend(site_rows)
        structure.extend(site_structure)
        acceptance[key] = {"observed_readout": observed_acceptance(readout), "native_modes": modes_acceptance(modes),
                           "common_a_current_gate": True, "fixed_observers": fixed_observers}
        tunings[key] = {"observed_readout": readout["tuning"], "native_modes": modes["tuning"],
                        "mode_tuning_mean_mse": modes.get("tuning_mean_mse"), "mode_pcg": modes["pcg"],
                        "mode_spectrum": modes["spectrum"], "mode_matrix_compile_cost": modes.get("matrix_compile_cost")}
        parameter_path = output / "site_parameters" / f"{key}.json"
        write_json(parameter_path, {"site": key, "identity": site["identity"],
                                    "observed_readout": readout["parameters"], "native_modes": modes["parameters"]})
        parameters["sites"][key] = {"path": str(parameter_path), "sha256": sha_file(parameter_path),
                                    "bytes": parameter_path.stat().st_size}
        torch.cuda.synchronize()
        event(output, "SITE_DONE", site=key, elapsed_seconds=round(time.monotonic() - site_start, 2),
              mode_status=modes["status"], query_rows=len(site_rows), structure_rows=len(site_structure),
              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device))
        del site, direction, panels, pair, readout, modes
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    selected = select_families(query, acceptance, archive["dev_sources"], list(site_identities), spec["local_admission"])
    write_json(output / "BINDINGS.json", {"scope_sha256": sha_file(scope_path), "parent_bindings": scope["bindings"],
               "checked_endpoints": len(references)*2, "full80_q1_locator_count": len(archive["strict_q1_full80"]),
               "checkpoints": archive["checkpoints"], "native_sources": scope["native_sources"],
               "sites": site_identities, "source_clusters": {name: row["source_cluster_id"] for name, row in bank.items()},
               "fold_identities": bind_fold_receipts(loaded_ids), "schedule": archive["schedule"],
               "historical_p64_role": "context_only_outside_local_gate", "nfe": 0, "d4": 0})
    write_json(output / "CAL_PARAMETERS.json", parameters)
    write_json(output / "CAL_TUNING.json", tunings)
    write_csv(output / "LOCAL_QUERY.csv", query)
    write_csv(output / "LOCAL_STRUCTURE.csv", structure)
    write_json(output / "BF16_ACCEPTANCE.json", {"scope_sha256": sha_file(scope_path), "runtime": runtime_info,
               "positional_scope": "P512 on six archived Wan positions", "sites": acceptance,
               "gate_witness": "actual native post_process with archived current BF16 gate on a synthetic residual addend",
               "nfe": 0, "d4": 0})
    write_json(output / "FAMILY_SELECTION.json", selected)
    event(output, "LOCAL_COMPLETE", next_stage=selected["next_stage"], primary=selected["primary"], nfe=0, d4=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    run(parser.parse_args().scope)
