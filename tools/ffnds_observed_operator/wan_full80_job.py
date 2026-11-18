"""Run one reviewed Wan full80 arm on one frozen source or all three."""

import argparse
import os
from pathlib import Path
import resource
import shutil
from time import monotonic

from icc_native_job import authorized_stage, build_pipeline, read_json, sha, write_json
from wan_comparison_job import bind_parameter_loader, bound_json, render

from ffnds_state_operator.native import tensor_sha
from ffnds_observed_operator.native_full80 import (
    ARMS, DEV3, L64_SITES, SITES, WIDTH, Full80Port, bind_full80_gate, row_ids,
)


STRICT_Q1_SHA = "dc2d2b2289ff119d432c5a347facc446601e5ef2526eb10c09f86f06df88e247"
STRICT_TRANSFORMER = "5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225"
STRICT_DENSE_NOISE = "ecf445d71318ab7dbed947c9f277b7054ef1e801faaaa70b9c0c6a6452dfc8a6"
ORIGINAL_BINDINGS = "8f7320c61dbafd44df0b1662235405de8961dd5c5d52d846c979fdb4179b1ab6"


def same_asset(left, right):
    return (left.get("path") or left.get("file"), left["sha256"]) == (
        right.get("path") or right.get("file"), right["sha256"])


def site_loader(scope, q1, torch):
    manifest = bound_json(scope["full80_parameters"])
    p0 = bound_json(scope["bindings"])
    strict = bound_json(scope["parameters"])["sites"]
    q1_refs = {(row["expert"], row["layer"]): row["artifact"] for row in strict}
    entries = manifest["sites"]
    needed = {f"{expert}{layer:02d}" for expert, layer in SITES}
    six = {f"{expert}{layer:02d}" for expert, layer in L64_SITES}
    if (manifest["status"] != "PASS" or set(entries) != needed or len(strict) != 80
            or set(q1_refs) != SITES or set(p0["sites"]) != six):
        raise RuntimeError("Expected complete CAL/Q1 full80 and the original six-site L64 binding")
    if p0["status"] != "PASS" or p0["l64_policy"] != "REUSE_ALL_SIX":
        raise RuntimeError("Original six-site L64 binding differs")
    family = "observed_readout" if scope["arm"].startswith("readout-") else "native_modes"

    def read_site(expert, layer):
        key = f"{expert}{layer:02d}"
        entry = entries[key]
        path = Path(entry["path"])
        if not path.is_absolute() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Unavailable or changed full80 parameter size at {key}")
        doc = bound_json(entry)
        identity = doc["identity"]
        if doc["site"] != key or not doc[family]:
            raise RuntimeError(f"Wrong site or unavailable frozen {family} at {key}")
        if (expert, layer) in L64_SITES:
            old = p0["sites"][key]
            if (old["bf16_bitwise_equal"] is not True or not same_asset(identity["old_params"], old["old_params"])
                    or not same_asset(identity["strict_q1"], q1_refs[expert, layer])):
                raise RuntimeError(f"Wrong frozen L64 identity at {key}")
        else:
            if (identity["base_role"] != "STRICT_Q1" or "l64_tensor_sha256" in identity
                    or not same_asset(identity["strict_q1"], q1_refs[expert, layer])):
                raise RuntimeError(f"Wrong strict Q1 identity at {key}")
        return doc[family], {**identity, "family_parameters": entry}

    # Bind every parameter file before committing this job's first denoising call.
    for expert, layer in sorted(SITES):
        _, identity = read_site(expert, layer)
        key = "old_params" if (expert, layer) in L64_SITES else "strict_q1"
        asset = identity[key]
        path = Path(asset["path"])
        if not path.is_absolute() or ("bytes" in asset and path.stat().st_size != asset["bytes"]):
            raise RuntimeError(f"Unavailable or changed full80 base artifact {path}")
        if sha(path) != asset["sha256"]:
            raise RuntimeError(f"Changed full80 base artifact {path}")

    def load(expert, layer, device):
        params, identity = read_site(expert, layer)
        if (expert, layer) in L64_SITES:
            old = identity["old_params"]
            if sha(old["path"]) != old["sha256"]:
                raise RuntimeError("Changed bound L64 tensor file")
            data = torch.load(old["path"], map_location="cpu", mmap=True, weights_only=True)
            a = data["A_L_BF16"]
            if tensor_sha(a) != identity["l64_tensor_sha256"]:
                raise RuntimeError("Changed bound deployed L64 tensor")
            del data
            a = a.to(device)
        else:
            a = q1(expert, layer, "cpu")
            if tensor_sha(a) != identity["strict_bf16_tensor_sha256"]:
                raise RuntimeError("Strict Q1 BF16 tensor differs from CAL deployment base")
            a = a.to(device)
        if a.shape != (WIDTH, WIDTH) or a.dtype != torch.bfloat16 or not bool(torch.isfinite(a).all()):
            raise RuntimeError("Invalid deployed full80 baseline")
        return a, params, identity

    return load


def weight_witness(torch):
    """Check the first live use of each expert/layer against CAL's BF16 bytes."""
    seen = set()

    def verify(expert, layer, phase, identity):
        if (expert, layer) in seen:
            return
        prefix = f"blocks.{layer}.ffn"
        for index, linear, m, n in ((0, phase.ffn_0, WIDTH, 13824), (2, phase.ffn_2, 13824, WIDTH)):
            weight = linear._get_actual_weight()
            bias = linear._get_actual_bias()
            if (weight.shape != (m, n) or weight.dtype != torch.bfloat16
                    or bias is None or bias.numel() != n or bias.dtype != weight.dtype):
                raise RuntimeError(f"Wrong live BF16 weight/bias geometry at {expert}/{prefix}.{index}")
            for suffix, value in (("weight", weight.T), ("bias", bias.reshape(-1))):
                name = f"{prefix}.{index}.{suffix}"
                if tensor_sha(value) != identity["ffn_tensor_sha256"][name]:
                    raise RuntimeError(f"Live BF16 tensor differs from CAL at {expert}/{name}")
        seen.add((expert, layer))

    return verify


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("scope_sha", nargs="?", default=os.environ.get("FFNDS_SCOPE_SHA256"))
    args = cli.parse_args()
    if not args.scope_sha:
        cli.error("scope_sha or FFNDS_SCOPE_SHA256 is required")
    stage = Path.cwd().resolve()
    scope, reference = authorized_stage(stage, args.scope_sha)
    sources = tuple(scope["sources"])
    if (scope["method"] != "WAN_FREE_GENERATION" or scope["arm"] not in ARMS
            or scope["force_dense"] is not False
            or sources not in (DEV3, *((source,) for source in DEV3))
            or (scope["max_d4"], scope["max_nfe"], scope["seed"], scope["retries"]) != (len(sources), 4 * len(sources), 42, 0)
            or scope["parameters"]["sha256"] != STRICT_Q1_SHA
            or scope["bindings"]["sha256"] != ORIGINAL_BINDINGS
            or scope["native_transformer_sha256"] != STRICT_TRANSFORMER
            or scope["dense_initial_noise"]["sha256"] != STRICT_DENSE_NOISE):
        raise RuntimeError("Execution scope differs from the frozen full80 design")
    canonical = scope["fixed64_token_ids"]
    probe, query = row_ids(canonical)

    import torch
    import runner_v2 as v2

    started = monotonic()
    limits = scope["resources"]

    def check_resources():
        if monotonic() - started > limits["soft_seconds"]:
            raise TimeoutError("Full80 soft deadline")
        if shutil.disk_usage(stage).free < limits["in_flight_disk_floor_gib"] * 1024 ** 3:
            raise RuntimeError("Full80 in-flight disk floor")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["host_memory_gib"] * 1024 ** 3:
            raise MemoryError("Full80 host RSS limit")
        if torch.cuda.max_memory_reserved() > limits["gpu_memory_gib"] * 1024 ** 3:
            raise MemoryError("Full80 GPU reserved-memory limit")

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
            "schema": "FFNDS_OBSERVED_FULL80_RECIPE_PARITY_V1", "scope_sha256": args.scope_sha,
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "strict": expected,
            "live": {"path": str(live_config), "sha256": live_sha},
            "exact_bytes_equal": live_sha == expected["sha256"], "nfe_at_check": 0,
        })
        if live_sha != expected["sha256"]:
            raise RuntimeError("Live effective configuration differs byte-for-byte from strict Q1")
        q1 = bind_parameter_loader(scope["parameters"], prior["checkpoints"], method="Q1r2")
        load = site_loader(scope, q1, torch)

        verify_weights = weight_witness(torch)

        def make(source):
            return Full80Port(output / f"{source}_cache", output / f"{source}_query",
                              source, scope["arm"], load, probe, query, verify_weights)

        def extra_binding(owner, port):
            return bind_full80_gate(owner, port, STRICT_TRANSFORMER)

        write_json(output / "BINDINGS.json", {
            "schema": "FFNDS_OBSERVED_FULL80_BINDINGS_V1", "scope_sha256": args.scope_sha,
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "arm": scope["arm"], "sources": list(sources),
            "strict_q1": scope["parameters"], "p0": scope["bindings"], "parameters": scope["full80_parameters"],
            "l64_sites": sorted(f"{expert}{layer:02d}" for expert, layer in L64_SITES),
            "canonical_global_p64_ids": canonical, "fixed_probe_ids": probe, "query_ids": query,
            "recipe_parity": {"path": str(parity), "sha256": sha(parity)},
            "checkpoints": prior["checkpoints"], "producer_sha256": scope["bundle_sha256"],
            "timing_includes_query_capture": True,
        })
        render(pipe, scope, output, conditions, make, check_resources,
               extra_binding=extra_binding, trace_flows=True)


if __name__ == "__main__":
    main()
