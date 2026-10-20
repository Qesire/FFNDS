"""Validate both genuinely reduced FFN GEMMs on actual native HiF4 operands."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import atomic_json, digest
from compare_dense_trajectory import same_tensor_bits
from packed_ffn_reference import forward, prepare
from run_ar2_restart import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    if args.output.exists():
        raise RuntimeError("Select fresh output")
    torch.set_num_threads(8)
    cache = json.loads((ROOT / ".runtime/hif4_rebuilt_cache.json").read_text())
    mirror = json.loads((ROOT / ".runtime/node_local_model_view.json").read_text())
    cap = json.loads(args.capture.read_text())
    masks = json.loads((ROOT / "config/ar2_frozen_random_masks.json").read_text())["manifest"]["masks"]
    report = {"status": "RUNNING", "capture_sha256": digest(args.capture), "cache_sha256": cache["sha256"],
              "source_sha256": {str(p): digest(p) for p in [Path(__file__), ROOT / "tools/packed_ffn_reference.py"]},
              "scope": "U through Q1/FFN0/GELU/Q2/gain/FFN2; gate tested, norm/residual timing excluded",
              "full_FFN_speed_acceptance": False, "closed_loop_video_acceptance": False, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for scope in [s for s in cap["scopes"] if s["block"] == 20 and s["step"] in [0, 2]]:
            if digest(scope["tensor_file"]) != scope["tensor_sha256"]:
                raise RuntimeError("Capture changed")
            with safe_open(scope["tensor_file"], framework="pt", device="cpu") as handle:
                t = {k: handle.get_tensor(k).cuda() for k in ["U", "V1", "Z", "H", "V2", "Y", "b2", "gate_broadcast"]}
            shard = next(s for s in cache["shards"] if s["expert"] == scope["expert"] and s["block"] == 20)
            if digest(shard["path"]) != shard["sha256"]:
                raise RuntimeError("Cache shard changed")
            payload = torch.load(shard["path"], map_location="cpu", weights_only=True, mmap=True)[scope["expert"]]
            w1, w2 = [payload[f"blocks.20.ffn.{i}"]["weight_qdq"].cuda() for i in [0, 2]]
            s1, s2 = [payload[f"blocks.20.ffn.{i}"]["smooth_scale"].cuda() for i in [0, 2]]
            src = next(r for r in mirror["files"] if f"/{scope['expert']}/" in r["view_path"] and r["filename"].endswith(".safetensors"))
            with safe_open(src["view_path"], framework="pt", device="cpu") as handle:
                b1 = handle.get_tensor("blocks.20.ffn.0.bias").cuda()
            b2, gate = t["b2"].reshape(-1), t["gate_broadcast"].reshape(1, 5120)
            v1 = qdq((t["U"].float() / s1).to(torch.bfloat16), force_fp32=True)
            assert same_tensor_bits(v1, t["V1"])
            z = F.linear(v1, w1, b1)
            assert same_tensor_bits(z, t["Z"])
            a = qdq((F.gelu(z, approximate="tanh").float() / s2).to(torch.bfloat16), force_fp32=True)
            assert same_tensor_bits(a, t["V2"])
            assert same_tensor_bits(F.linear(a, w2, b2), t["Y"])
            for mask in masks:
                packed = prepare(w1, w2, s1, s2, b1, b2, mask["kept_group_ids"])
                channels = packed["channels"]
                z_pack = F.linear(v1, packed["w1"], packed["b1"])
                assert same_tensor_bits(z_pack, z.index_select(1, channels)), "Reduced FFN0 changed retained neurons"
                for gain in [1.0, 1.3]:
                    candidate = forward(t["U"], packed, gain=gain) * gate
                    retained = (a.index_select(1, channels).float() * gain).to(torch.bfloat16)
                    masked = torch.zeros_like(a).index_copy_(1, channels, retained)
                    reference = F.linear(masked, w2, b2) * gate
                    error = (candidate.double() - reference.double())
                    assert same_tensor_bits(candidate, reference), "Native packed two-GEMM result differs"
                    report["cases"].append({"step": scope["step"], "mask": mask["mask_id"], "gain": gain,
                        "rows": len(a), "both_packed_GEMMs": [list(packed["w1"].shape), list(packed["w2"].shape)],
                        "FFN0_main_GEMM_density": .5, "FFN2_main_GEMM_density": .5,
                        "full_native_replay_bitwise": True, "retained_FFN0_Z_bitwise": True,
                        "packed_vs_full_masked_output_bitwise": True, "max_abs": float(error.abs().max()), "bias_once": True})
            print(f"PASS step{scope['step']} packed FFN0+FFN2", flush=True)
        assert len(report["cases"]) == 16
        report["status"] = "PASS_BOTH_PACKED_FFN_GEMMS_NUMERICAL_ONLY"
    except BaseException as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
