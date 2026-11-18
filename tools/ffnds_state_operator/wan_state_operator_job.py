"""One reviewed three-source arm of the state-operator Wan production cohort."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
from time import monotonic

from icc_native_job import authorized_stage, build_pipeline, read_json, sha, write_json
from wan_comparison_job import bind_parameter_loader, bound_json, render
from wan_ffn_port import source_digest

from ffnds_state_operator.native import ARMS, SITES, SOURCES, bind_gate_witness, make_port


STRICT_Q1_SHA = "dc2d2b2289ff119d432c5a347facc446601e5ef2526eb10c09f86f06df88e247"
STRICT_TRANSFORMER = "5fa53d102aed4e2793b4d639da65a71e1e63061d9b2c35574db1b4ac2256c225"
STRICT_DENSE_NOISE = "ecf445d71318ab7dbed947c9f277b7054ef1e801faaaa70b9c0c6a6452dfc8a6"


def site_loader(scope, torch):
    from ffnds_state_operator.fit import load_q1_math

    fit = bound_json(scope["fit"])
    policy = bound_json(fit["bindings"])
    if fit["schema"] != "FFNDS_STATE_OPERATOR_FIT_V1" or scope["bindings"] != fit["bindings"]:
        raise RuntimeError("fit is outside the frozen P0 identity binding")
    entries = {(row["expert"], row["layer"]): row["params"] for row in fit["sites"].values()}
    if set(entries) != SITES or len(fit["sites"]) != 6:
        raise RuntimeError("fit requires all six distinct frozen sites")
    binding_sites = policy["sites"]

    def load(expert, layer):
        entry = entries[expert, layer]
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("fit parameter file differs from frozen artifact")
        params = torch.load(entry["path"], map_location="cpu", mmap=True, weights_only=True)
        q1_math = load_q1_math(binding_sites[f"{expert}{layer:02d}"], policy["l64_policy"], "cpu")
        return params, q1_math

    return load


def verify_p1(scope):
    spec = scope["bf16_acceptance"]
    receipt = bound_json(spec["result"])
    run = bound_json(spec["run"])
    exited = bound_json(spec["exit"])
    sites = receipt["sites"]
    expected = {(expert, layer, arm) for expert, layer in SITES for arm in ARMS}
    actual = {(row["expert"], row["layer"], row["arm"]) for row in sites}
    if (receipt["schema"] != "FFNDS_STATE_OPERATOR_BF16_V1" or receipt["status"] != "PASS"
            or receipt["fit"] != scope["fit"] or receipt["strict_q1"] != scope["parameters"]
            or receipt["native_transformer_sha256"] != STRICT_TRANSFORMER
            or run["scope_sha256"] != spec["scope_sha256"] or receipt["scope_sha256"] != spec["scope_sha256"]
            or str(receipt["slurm_job_id"]) != str(spec["slurm_job_id"])
            or (receipt["nfe"], receipt["d4"], exited["exit_code"]) != (0, 0, 0)
            or str(exited["slurm_job_id"]) != str(spec["slurm_job_id"])
            or len(sites) != 24 or actual != expected
            or not all(row["exact_deployed_order"] is True and row["native_gate_write_bitwise_equal"] is True
                       and row["q1_shared_arithmetic_bitwise_equal"] is True for row in sites)):
        raise RuntimeError("P1 BF16 acceptance differs from the reviewed execution evidence")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("scope_sha")
    args = cli.parse_args()
    stage = Path.cwd().resolve()
    scope, reference = authorized_stage(stage, args.scope_sha)
    if (scope["method"] != "WAN_FREE_GENERATION" or scope["arm"] not in ARMS
            or scope["force_dense"] is not False or tuple(scope["sources"]) != SOURCES
            or (scope["max_d4"], scope["max_nfe"], scope["seed"], scope["retries"]) != (3, 12, 42, 0)
            or scope["parameters"]["sha256"] != STRICT_Q1_SHA
            or scope["native_transformer_sha256"] != STRICT_TRANSFORMER
            or scope["dense_initial_noise"]["sha256"] != STRICT_DENSE_NOISE):
        raise RuntimeError("scope differs from the frozen state-operator P2 design")
    verify_p1(scope)

    import torch
    import runner_v2 as v2

    started = monotonic()
    limits = scope["resources"]

    def check_resources():
        if monotonic() - started > limits["soft_seconds"]:
            raise TimeoutError("state-operator soft deadline")
        if shutil.disk_usage(stage).free < limits["in_flight_disk_floor_gib"] * 1024 ** 3:
            raise RuntimeError("state-operator in-flight disk floor")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["host_memory_gib"] * 1024 ** 3:
            raise MemoryError("state-operator host RSS limit")
        if torch.cuda.max_memory_reserved() > limits["gpu_memory_gib"] * 1024 ** 3:
            raise MemoryError("state-operator GPU reserved-memory limit")

    with torch.inference_mode():
        rows = read_json(reference / "v2_conditions.json")
        v2.verify_conditions(rows)
        conditions = {row["record_id"]: row for row in rows}
        pipe, output, prior = build_pipeline(stage, scope, reference)
        q1 = bind_parameter_loader(scope["parameters"], prior["checkpoints"], method="Q1r2")
        load = site_loader(scope, torch)

        def make(source):
            return make_port(output / f"{source}_cache", scope["arm"], q1, load)

        def extra_binding(owner, port):
            return bind_gate_witness(owner, port, STRICT_TRANSFORMER, source_digest=source_digest)

        write_json(output / "BINDINGS.json", {
            "scope_sha256": args.scope_sha, "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "arm": scope["arm"], "strict_q1": scope["parameters"], "fit": scope["fit"],
            "p0": scope["bindings"], "p1": scope["bf16_acceptance"],
            "checkpoints": prior["checkpoints"], "producer_sha256": scope["bundle_sha256"],
        })
        render(pipe, scope, output, conditions, make, check_resources,
               extra_binding=extra_binding, trace_flows=True)


if __name__ == "__main__":
    main()
