"""Exercise BF16 GEMM and torch SDPA on the selected GPU; no performance claim."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; choose a new path")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1, "Use the per-GPU remote wrapper"
    assert torch.cuda.is_bf16_supported()
    torch.manual_seed(20261020)
    # Dyadic inputs make the exact BF16 result check independent of TF32.
    a = torch.randint(-3, 4, (64, 128)).float() / 8
    b = torch.randint(-3, 4, (128, 64)).float() / 8
    actual = a.cuda().bfloat16() @ b.cuda().bfloat16()
    expected = (a.double() @ b.double()).bfloat16()
    assert torch.equal(actual.cpu(), expected)
    q, k, v = [torch.randn(2, 4, 128, 64).bfloat16() for _ in range(3)]
    attn = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda())
    ref = (torch.softmax((q.double() @ k.double().transpose(-2, -1)) / 8, dim=-1) @ v.double()).float()
    torch.testing.assert_close(attn.cpu().float(), ref, atol=0.008, rtol=0.02)
    torch.cuda.synchronize()
    result = {"status": "PASS", "scope": "synthetic BF16 GEMM and SDPA numerical validation only",
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": torch.cuda.get_device_name(),
              "compute_capability": list(torch.cuda.get_device_capability()),
              "bf16_gemm_exact": True, "sdpa_finite": bool(torch.isfinite(attn).all()),
              "sdpa_max_abs_error_vs_fp64": float((attn.cpu().float() - ref).abs().max()),
              "peak_memory_bytes": torch.cuda.max_memory_allocated()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
