"""Scoped P1 BF16 acceptance using archived native P512 tensors, without Wan generation."""

import argparse
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from icc_native_job import authorized_stage, read_json, sha, write_json
from wan_comparison_job import bound_json

from ffnds_state_operator.bf16_replay import bitwise_equal, check_panel, compile_endpoint, replay_hit, residual
from ffnds_state_operator import fit
from ffnds_state_operator.native import ARMS, SITES, tensor_sha
from wan_state_operator_job import STRICT_Q1_SHA, STRICT_TRANSFORMER, site_loader


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("scope_sha")
    args = cli.parse_args()
    stage = Path.cwd().resolve()
    scope, reference = authorized_stage(stage, args.scope_sha)
    if (scope["method"] not in ("STATE_OPERATOR_BF16_REPLAY", "STATE_OPERATOR_P1")
            or scope["max_nfe"] != 0 or scope["max_d4"] != 0
            or scope["parameters"]["sha256"] != STRICT_Q1_SHA
            or scope["native_transformer_sha256"] != STRICT_TRANSFORMER):
        raise RuntimeError("scope does not authorize the frozen P1-only replay")
    import torch
    import runner_v2 as v2

    prior = read_json(reference / "scope.json")
    v2.verify_pipeline_binding(prior)
    for path in reversed(prior["python_import_paths"]):
        sys.path.insert(0, path)
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

    original_post = WanTransformerInfer.post_process
    if sha(inspect.getfile(original_post)) != STRICT_TRANSFORMER:
        raise RuntimeError("native post_process source changed")
    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != "NVIDIA A100-SXM4-80GB"
            or torch.__version__ != prior["runtime_versions"]["torch"]):
        raise RuntimeError("P1 BF16 needs the frozen A100 numeric runtime")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    infer_dtype, sensitive_dtype = GET_DTYPE(), GET_SENSITIVE_DTYPE()
    if (infer_dtype, sensitive_dtype) != (torch.bfloat16, torch.bfloat16):
        raise RuntimeError("actual LightX2V inference and sensitive dtypes must be BF16")
    effective_spec = scope.get("strict_effective_config")
    config = bound_json(effective_spec) if effective_spec else {}
    if effective_spec:
        if (config.get("use_bfloat16") is not True or config.get("clean_cuda_cache", False)
                or any(config.get(key) != expected
                       for key, expected in prior["required_runtime_config"].items())):
            raise RuntimeError("strict effective pipeline no longer matches the BF16 replay contract")
    host = SimpleNamespace(sensitive_layer_dtype=sensitive_dtype, infer_dtype=infer_dtype,
                           clean_cuda_cache=config.get("clean_cuda_cache", False))
    strict = bound_json(scope["parameters"])
    bind = bound_json(scope["bindings"])
    if bind["schema"] != fit.BINDINGS_SCHEMA or bind["status"] != "PASS":
        raise RuntimeError("P1 requires the checked P0 identity output")
    strict_sites = {(row["expert"], row["layer"]): row["artifact"] for row in strict["sites"]}
    for expert, layer in SITES:
        bound = bind["sites"][f"{expert}{layer:02d}"]["strict_q1"]
        expected = strict_sites[expert, layer]
        if (bound["path"], bound["sha256"]) != (expected["path"], expected["sha256"]):
            raise RuntimeError("P0 and strict Q1 site artifact identity differs")
    output = stage / "output"
    output.mkdir()
    fit_scope = scope
    if scope["method"] == "STATE_OPERATOR_P1":
        if "fit" in scope:
            raise RuntimeError("combined P1 does not accept a preexisting fit")
        for name in ("fit.py", "features.py", "analysis.py"):
            path = Path(fit.__file__).with_name(name).resolve()
            relative = str(path.relative_to(stage))
            if scope["bundle_sha256"][relative] != sha(path):
                raise RuntimeError(f"fit source differs from reviewed bundle: {name}")
        device = torch.device("cuda", torch.cuda.current_device())
        limit = scope["resources"]["gpu_memory_gib"] * 1024 ** 3
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total), device)
        if bind["l64_policy"] == "REFIT_ALL_SIX":
            legacy_root = "reports/20261114/g20_two_track_core/A_Q1_RESIDUAL/v1/source"
            for name in ("a_q1_residual_repr.py", "ridge_fit_core.py"):
                relative = f"{legacy_root}/{name}"
                if sha(stage / relative) != scope["bundle_sha256"][relative]:
                    raise RuntimeError(f"legacy fit source differs from reviewed bundle: {name}")
        fit.run(fit.verified(scope["bindings"]).resolve(), output / "fit", device)
        torch.cuda.synchronize()
        path = output / "fit" / "FIT_MANIFEST.json"
        fit_scope = {**scope, "fit": {"path": str(path), "sha256": sha(path)}}
        manifest = bound_json(fit_scope["fit"])
        for filename, digest in manifest["producer"].items():
            if filename == "legacy_basis_and_center":
                filename = "reports/20261114/g20_two_track_core/A_Q1_RESIDUAL/v1/source/a_q1_residual_repr.py"
            else:
                filename = f"ffnds_state_operator/{filename}"
            if scope["bundle_sha256"][filename] != digest or sha(stage / filename) != digest:
                raise RuntimeError(f"fit producer differs from reviewed bundle: {filename}")
        for filename, digest in manifest["derived"].items():
            if sha(path.parent / filename) != digest:
                raise RuntimeError(f"fit-derived artifact digest differs: {filename}")
    load = site_loader(fit_scope, torch)
    by_site = {}
    for row in bind["pairs"]:
        if row["split"] == "DEV":
            by_site.setdefault((row["expert"], row["layer"]), {})[row["source"]] = row
    if set(by_site) != SITES or any(set(pairs) != set(bind["dev_sources"]) for pairs in by_site.values()):
        raise RuntimeError("P1 BF16 replay requires all DEV11 panels at all six sites")
    checkpoint_by_expert = {
        expert: fit.verified(bind["checkpoints"][expert]) for expert in ("high", "low")
    }
    print("Replaying frozen native DEV11 panels in BF16", flush=True)
    results = []
    with torch.inference_mode():
        for expert, layer in sorted(SITES):
            p, q1_math = load(expert, layer)
            site = bind["sites"][f"{expert}{layer:02d}"]
            q1 = torch.load(fit.verified(site["strict_q1"]), map_location="cpu", mmap=True,
                            weights_only=True)["A_q1_f32"].to("cuda", dtype=torch.bfloat16)
            w1, bias, *_ = fit.load_weights(checkpoint_by_expert[expert], layer, "cpu")
            endpoints = {arm: compile_endpoint(arm, p, q1_math, "cuda", w1=w1, b1=bias) for arm in ARMS}
            arm_records = {arm: [] for arm in ARMS}
            if bind["l64_policy"] == "REUSE_ALL_SIX":
                old = torch.load(fit.verified(site["old_params"]), map_location="cpu", weights_only=True)
                if not bitwise_equal(endpoints["L0"].matrix, old["A_L_BF16"].to("cuda")):
                    raise RuntimeError("L0 differs from the old accepted BF16 L64 deployment")
            for source, pair_ref in by_site[expert, layer].items():
                pair = fit.load_pair(pair_ref, "cuda")
                a, u, y0, y1 = (pair[key].bfloat16().contiguous() for key in ("u0", "u1", "y0", "y1"))
                gate = pair["g"].bfloat16()
                gate_ref = torch.load(fit.verified(pair_ref["anchor"]), map_location="cpu", weights_only=True)["gate"]
                if gate_ref.dtype != torch.bfloat16 or gate_ref.shape != (1, 1, 5120):
                    raise RuntimeError("native anchor gate geometry changed")
                gate0 = gate_ref.to("cuda")
                native_anchor = original_post(host, a.clone(), y0, gate0)
                if not bitwise_equal(native_anchor, a + y0 * gate0.squeeze()):
                    raise RuntimeError("native BF16 anchor gate write changed")
                for arm, endpoint in endpoints.items():
                    check = check_panel(endpoint, a, u, y0, gate, q1=q1,
                                        original_post=original_post, host=host)
                    hit = replay_hit(endpoint(u), residual(y0, endpoint(a)))
                    target = (y1.float() - y0.float()) * gate.float()
                    error = ((hit.float() - y0.float()) * gate.float() - target).square().mean()
                    baseline = target.square().mean().item()
                    mse = error.item()
                    arm_records[arm].append({"source": source, **check,
                                             "anchor_semantics": "archived_native_pre_gate_passthrough",
                                             "anchor_gate_write_bitwise_equal": True,
                                             "anchor_gate_sha256": tensor_sha(gate_ref),
                                             "hit_gate_sha256": tensor_sha(gate),
                                             "anchor_pre_gate_sha256": tensor_sha(y0),
                                             "current_gate_gated_mse": mse,
                                             "current_gate_target_mse": baseline,
                                             "current_gate_relative_risk": mse / baseline if baseline > 0 else None})
            for arm, records in arm_records.items():
                results.append({"expert": expert, "layer": layer, "arm": arm,
                                "exact_deployed_order": all(row["exact_deployed_order"] for row in records),
                                "native_gate_write_bitwise_equal": all(row["native_gate_write_bitwise_equal"] for row in records),
                                "q1_shared_arithmetic_bitwise_equal": all(row["q1_shared_arithmetic_bitwise_equal"] for row in records),
                                "pairs": records})
    write_json(output / "BF16_ACCEPTANCE.json", {
        "schema": "FFNDS_STATE_OPERATOR_BF16_V1", "status": "PASS", "fit": fit_scope["fit"],
        "strict_q1": scope["parameters"], "native_transformer_sha256": STRICT_TRANSFORMER,
        "producer_sha256": scope["bundle_sha256"], "scope_sha256": args.scope_sha,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "nfe": 0, "d4": 0, "sites": results,
        "native_host": {"infer_dtype": str(infer_dtype), "sensitive_dtype": str(sensitive_dtype),
                        "strict_effective_config": effective_spec,
                        "clean_cuda_cache": host.clean_cuda_cache},
        "signed_acceptance": {"endpoint_cache_hit_writeback_bitwise": True,
                              "zero_delta": "BF16 cache and output rounding only",
                              "native_residual": "synthetic addend, real native gate and pre-gate endpoint"},
    })


if __name__ == "__main__":
    main()
