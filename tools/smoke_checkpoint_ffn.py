"""Verify restored native FFN callables using real D weights and synthetic U.

This checks layout, GELU(tanh), bias, gate, and residual arithmetic on selected
blocks. Inputs are synthetic, so this is not a captured-scope experiment, a
sparsity result, a video check, or a performance benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-manifest", type=Path, default=ROOT / ".runtime/models.json")
    ap.add_argument("--expert", choices=["high", "low"], default="high")
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 20])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; choose a new path")
    report = {"status": "RUNNING", "scope": "real checkpoint FFN operators with synthetic U/gate/residual only",
              "arm": "D-BF16", "expert": f"{args.expert}_noise_model", "checks": []}
    try:
        manifest = json.loads(args.model_manifest.read_text())
        contract_path = ROOT / "upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json"
        contract = json.loads(contract_path.read_text())
        filename = contract[f"{args.expert}_noise_filename"]
        record = next(row for row in manifest["files"] if row["filename"] == filename)
        assert record["sha256"] == contract[f"{args.expert}_noise_sha256"]
        checkpoint = Path(record["path"])
        assert checkpoint.stat().st_size == record["bytes"]
        report.update(checkpoint=record, torch=torch.__version__, device=torch.cuda.get_device_name())
        from lightx2v.common.ops.mm.mm_weight import MMWeight
        from lightx2v.common.ops.mm import mm_weight
        report["mmweight_source_sha256"] = digest(mm_weight.__file__)
        report["script_sha256"] = digest(__file__)
        report["contract_sha256"] = digest(contract_path)
        torch.manual_seed(20261020)
        with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
            for block in args.blocks:
                names = {part: f"blocks.{block}.ffn.{part}" for part in [0, 2]}
                values = {f"{name}.{kind}": tensors.get_tensor(f"{name}.{kind}").to("cuda")
                          for name in names.values() for kind in ["weight", "bias"]}
                mm0 = MMWeight(f"{names[0]}.weight", f"{names[0]}.bias")
                mm2 = MMWeight(f"{names[2]}.weight", f"{names[2]}.bias")
                # The native loader transposes checkpoint [out,in] to [in,out].
                expected0, bias0 = values[f"{names[0]}.weight"], values[f"{names[0]}.bias"]
                expected2, bias2 = values[f"{names[2]}.weight"], values[f"{names[2]}.bias"]
                mm0.load(dict(values))
                mm2.load(dict(values))
                assert list(mm0.weight.shape) == [5120, 13824]
                assert list(mm2.weight.shape) == [13824, 5120]
                u = torch.randn(32, 5120, dtype=torch.bfloat16, device="cuda")
                gate = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda") * .1
                residual = torch.randn_like(u)
                cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                z = mm0.apply(u)
                z_ref = F.linear(u, expected0, bias0)
                hidden = F.gelu(z, approximate="tanh")
                y = mm2.apply(hidden)
                y_ref = F.linear(F.gelu(z_ref, approximate="tanh"), expected2, bias2)
                delta = y * gate
                result = residual.clone()
                result.add_(delta)
                expected_result = residual + y_ref * gate
                zero = mm2.apply(torch.zeros_like(hidden))
                checks = {"ffn0_equal": torch.equal(z, z_ref), "ffn2_equal": torch.equal(y, y_ref),
                          "gate_residual_equal": torch.equal(result, expected_result),
                          "zero_hidden_returns_bias_once": torch.equal(zero, bias2.expand_as(zero)),
                          "cpu_rng_unchanged": torch.equal(cpu_rng, torch.get_rng_state()),
                          "cuda_rng_unchanged": torch.equal(cuda_rng, torch.cuda.get_rng_state()),
                          "all_finite": bool(torch.isfinite(result).all())}
                report["checks"].append({"block": block, "rows": len(u), **checks,
                    "W1_runtime_shape": list(mm0.weight.shape), "W2_runtime_shape": list(mm2.weight.shape),
                    "W2_bias_nonzero_channels": int(torch.count_nonzero(bias2)),
                    "ffn2_max_abs_difference": float((y.float() - y_ref.float()).abs().max())})
                assert all(checks.values()), checks
                del values, mm0, mm2, expected0, expected2, bias0, bias2
        torch.cuda.synchronize()
        report.update(status="PASS", peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
