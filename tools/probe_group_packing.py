"""Real-input native HiF4 group-packing/FFN2 probe, not a full FFN benchmark."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import atomic_json, digest
from compare_dense_trajectory import same_tensor_bits
from run_ar2_restart import tensor_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    root = Path(__file__).resolve().parents[1]
    capture = json.loads(args.capture.read_text())
    meta = json.loads((args.capture.parent.parent / "run_meta.json").read_text())
    cache = json.loads(Path(meta["quantized_cache_identity"]["manifest_path"]).read_text())
    masks = json.loads((root / "config/ar2_frozen_random_masks.json").read_text())
    groups = masks["manifest"]["masks"][0]["kept_group_ids"]
    channels = torch.tensor([g * 64 + j for g in groups for j in range(64)], device="cuda")
    report = {"status": "RUNNING", "purpose": "Q2 group commutation and FFN2 packing only",
              "full_FFN_speed_acceptance": False, "FFN0_reduction_tested": False,
              "capture_sha256": digest(args.capture), "cache_sha256": cache["sha256"], "cases": []}
    if args.output.exists():
        raise RuntimeError("Select a fresh probe output")
    try:
        for scope in [s for s in capture["scopes"] if s["block"] == 20 and s["step"] in [0, 2]]:
            if digest(scope["tensor_file"]) != scope["tensor_sha256"]:
                raise RuntimeError("Captured tensor changed")
            with safe_open(scope["tensor_file"], framework="pt", device="cpu") as handle:
                tensors = {k: handle.get_tensor(k).cuda() for k in ["Q2_input", "V2", "b2", "gate_broadcast"]}
            qq, vv, bias, gate = [tensors[k] for k in ["Q2_input", "V2", "b2", "gate_broadcast"]]
            bias, gate = (None if bias.numel() == 0 else bias.reshape(-1)), gate.reshape(1, 5120)
            full_q = qdq(qq, force_fp32=True)
            packed_q = qdq(qq.index_select(1, channels).contiguous(), force_fp32=True)
            assert same_tensor_bits(full_q, vv), "Sampled native Q2 replay differs"
            assert same_tensor_bits(packed_q, vv.index_select(1, channels)), "Whole-group packing does not commute"
            arbitrary = torch.arange(0, 13824, 2, device="cuda")
            changed = qdq(qq.index_select(1, arbitrary).contiguous(), force_fp32=True) != vv.index_select(1, arbitrary)
            rate = float(changed.float().mean())
            assert rate > 0, "Expected a non-group-preserving counterexample"
            shard = next(s for s in cache["shards"] if s["expert"] == scope["expert"] and s["block"] == 20)
            assert digest(shard["path"]) == shard["sha256"]
            payload = torch.load(shard["path"], map_location="cpu", weights_only=True, mmap=True)
            source = payload[scope["expert"]]["blocks.20.ffn.2"]["weight_qdq"]
            assert tensor_sha(source) == scope["effective_W2"]["runtime_operand_sha256"]
            weight = source.cuda()
            packed_weight = weight.index_select(1, channels).contiguous()
            masked = torch.zeros_like(vv).index_copy_(1, channels, packed_q)
            dense64 = F.linear(masked.double(), weight.double(), None if bias is None else bias.double()) * gate.double()
            packed64 = F.linear(packed_q.double(), packed_weight.double(), None if bias is None else bias.double()) * gate.double()
            assert torch.allclose(dense64, packed64, rtol=1e-12, atol=1e-12), "Packed W2/bias/layout mismatch"
            dense_native = F.linear(masked, weight, bias) * gate
            packed_native = F.linear(packed_q, packed_weight, bias) * gate
            zero = F.linear(torch.zeros_like(packed_q), packed_weight, bias) * gate
            expected_zero = torch.zeros_like(zero) if bias is None else (bias * gate).expand_as(zero)
            assert same_tensor_bits(zero, expected_zero), "Native bias was not retained exactly once"
            error = packed_native.double() - dense_native.double()
            report["cases"].append({"step": scope["step"], "expert": scope["expert"], "rows": len(vv),
                "native_Q2_replay_bitwise": True, "whole_group_QDQ_pack_bitwise": True,
                "arbitrary_channel_pack_changed_fraction": rate, "FP64_packed_projection_equal": True,
                "BF16_packed_vs_masked_relative_L2": float(error.norm() / dense_native.double().norm()),
                "BF16_packed_vs_masked_max_abs": float(error.abs().max()), "bias_once": True,
                "actual_packed_W2_shape": list(packed_weight.shape), "FFN2_main_GEMM_density": .5})
        assert len(report["cases"]) == 2
        report["status"] = "PASS_GROUP_QDQ_AND_FFN2_PACKING_ONLY"
    except BaseException as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
