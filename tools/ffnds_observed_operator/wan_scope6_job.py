"""Run one freshly reviewed six-site diagnostic arm on Wan 0392, seed 42."""

import argparse
import inspect
import os
from pathlib import Path
import resource
import shutil
import sys
from time import monotonic

from icc_native_job import authorized_stage, build_pipeline, read_json, sha, write_json
from wan_comparison_job import bind_parameter_loader, bound_json, render
from wan_ffn_comparator_ports import ComparatorPort
from wan_ffn_port import bind_native_ffn

from ffnds_observed_operator.native_full80 import L64_SITES, bind_full80_gate, row_ids
from ffnds_observed_operator.native_scope6 import ARMS, GP64, GP64_K, SOURCE, Scope6Port
from wan_full80_job import (
    ORIGINAL_BINDINGS, STRICT_DENSE_NOISE, STRICT_Q1_SHA, STRICT_TRANSFORMER, site_loader, weight_witness,
)


HISTORICAL_GP64_SCOPE_SHA = "abcbf252c09aafcf4df89d8bcd90addd46f0d8bedb75c8bfbad9ec0628595e0c"


def historical_binding(scope):
    spec = scope["global_p64_scope"]
    if spec["sha256"] != HISTORICAL_GP64_SCOPE_SHA:
        raise RuntimeError("Unexpected historical Global-P64 publication scope")
    historical = bound_json(spec)
    if (historical["scope_id"] != "FFNDS_STRICT_WAN_MAIN_GP64_01_V1"
            or historical["arm"] != "Global-P64" or historical["seed"] != 42
            or historical["force_dense"] is not False or SOURCE not in historical["sources"]
            or scope["fixed64_token_ids"] != historical["fixed64_token_ids"]
            or scope["dense_initial_noise"]["sha256"] != historical["dense_initial_noise"]["sha256"]
            or any(scope[key] != historical[key] for key in
                   ("schedule", "scheduler_sha256", "native_transformer_sha256"))):
        raise RuntimeError("Scope6 differs from the historical Global-P64 recipe or canonical probe IDs")
    for function, name in ((ComparatorPort.__call__, "wan_ffn_comparator_ports.py"),
                           (bind_native_ffn, "wan_ffn_port.py")):
        observed = sha(inspect.getfile(inspect.unwrap(function)))
        if observed != historical["bundle_sha256"][name] or observed != scope["bundle_sha256"][name]:
            raise RuntimeError(f"Historical Global-P64 source differs at {name}")
    return historical


def load_global_action(stage, historical, scope, torch):
    sys.path.insert(0, str(stage / "bws"))
    from chipmunk_fullm_proxy_math import rms_score, selection
    from chipmunk_wan_packed_pregate import packed_pregate

    for function, name in ((rms_score, "chipmunk_fullm_proxy_math.py"),
                           (selection, "chipmunk_fullm_proxy_math.py"),
                           (packed_pregate, "chipmunk_wan_packed_pregate.py")):
        path = stage / "bws" / name
        location = Path(inspect.getfile(function)).resolve()
        digest = historical["bws_sources_sha256"][name]
        if (location != path or sha(path) != digest
                or scope["bundle_sha256"][f"bws/{name}"] != digest):
            raise RuntimeError(f"Unexpected Global-P64 packing or scoring source at {name}")
    canonical = torch.tensor(historical["fixed64_token_ids"], device="cuda", dtype=torch.long)
    return canonical, rms_score, selection, packed_pregate


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("scope_sha", nargs="?", default=os.environ.get("FFNDS_SCOPE_SHA256"))
    args = cli.parse_args()
    if not args.scope_sha:
        cli.error("scope_sha or FFNDS_SCOPE_SHA256 is required")
    stage = Path.cwd().resolve()
    scope, reference = authorized_stage(stage, args.scope_sha)
    if (scope["method"] != "WAN_FREE_GENERATION" or scope["arm"] not in ARMS
            or scope["force_dense"] is not False or scope["sources"] != [SOURCE]
            or (scope["max_d4"], scope["max_nfe"], scope["seed"], scope["retries"]) != (1, 4, 42, 0)
            or scope["parameters"]["sha256"] != STRICT_Q1_SHA
            or scope["bindings"]["sha256"] != ORIGINAL_BINDINGS
            or scope["native_transformer_sha256"] != STRICT_TRANSFORMER
            or scope["dense_initial_noise"]["sha256"] != STRICT_DENSE_NOISE):
        raise RuntimeError("Execution scope differs from the fixed six-site diagnostic")
    historical = historical_binding(scope)
    canonical = historical["fixed64_token_ids"]
    probe, query = row_ids(canonical)

    import torch
    import runner_v2 as v2

    started = monotonic()
    limits = scope["resources"]

    def check_resources():
        if monotonic() - started > limits["soft_seconds"]:
            raise TimeoutError("Six-site soft deadline")
        if shutil.disk_usage(stage).free < limits["in_flight_disk_floor_gib"] * 1024 ** 3:
            raise RuntimeError("Six-site in-flight disk floor")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["host_memory_gib"] * 1024 ** 3:
            raise MemoryError("Six-site host RSS limit")
        if torch.cuda.max_memory_reserved() > limits["gpu_memory_gib"] * 1024 ** 3:
            raise MemoryError("Six-site GPU reserved-memory limit")

    with torch.inference_mode():
        rows = read_json(reference / "v2_conditions.json")
        v2.verify_conditions(rows)
        conditions = {row["record_id"]: row for row in rows}
        expected = scope["strict_effective_config"]
        bound_json(expected)
        pipe, output, prior = build_pipeline(stage, scope, reference)
        live_config = output / "EFFECTIVE_CONFIG.json"
        live_sha = sha(live_config)
        parity = output / "RECIPE_PARITY.json"
        write_json(parity, {
            "schema": "FFNDS_OBSERVED_SCOPE6_RECIPE_PARITY_V1", "scope_sha256": args.scope_sha,
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "strict": expected,
            "live": {"path": str(live_config), "sha256": live_sha},
            "exact_bytes_equal": live_sha == expected["sha256"], "nfe_at_check": 0,
        })
        if live_sha != expected["sha256"]:
            raise RuntimeError("Live effective configuration differs byte-for-byte from strict Q1")
        q1 = bind_parameter_loader(scope["parameters"], prior["checkpoints"], method="Q1r2")
        candidate_sites = sorted(f"{expert}{layer:02d}" for expert, layer in L64_SITES)
        gp64, gp64_identity, load = None, None, None
        if scope["arm"] == GP64:
            gp64 = load_global_action(stage, historical, scope, torch)
            original = bound_json(scope["bindings"])
            gp64_identity = {key: {
                "ffn_tensor_sha256": original["sites"][key]["ffn_tensor_sha256"],
                "global_p64_scope": scope["global_p64_scope"], "selected_k": GP64_K,
            } for key in candidate_sites}
        else:
            load = site_loader(scope, q1, torch)
        verify_weights = weight_witness(torch)

        def make(source):
            return Scope6Port(output / f"{source}_cache", output / f"{source}_query",
                              scope["arm"], q1, load, probe, query, verify_weights,
                              gp64=gp64, gp64_identity=gp64_identity)

        def extra_binding(owner, port):
            return bind_full80_gate(owner, port, STRICT_TRANSFORMER)

        write_json(output / "BINDINGS.json", {
            "schema": "FFNDS_OBSERVED_SCOPE6_BINDINGS_V1", "scope_sha256": args.scope_sha,
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "arm": scope["arm"], "sources": scope["sources"],
            "strict_q1": scope["parameters"], "p0": scope["bindings"], "parameters": scope["full80_parameters"],
            "candidate_sites": candidate_sites, "l64_sites": candidate_sites if scope["arm"] != GP64 else [],
            "strict_q1_sites_outside_candidate": 74, "global_p64_scope": scope["global_p64_scope"],
            "canonical_global_p64_ids": canonical, "fixed_probe_ids": probe, "query_ids": query,
            "recipe_parity": {"path": str(parity), "sha256": sha(parity)},
            "checkpoints": prior["checkpoints"], "producer_sha256": scope["bundle_sha256"],
            "timing_includes_query_capture": True,
        })
        render(pipe, scope, output, conditions, make, check_resources,
               extra_binding=extra_binding, trace_flows=True)


if __name__ == "__main__":
    main()
