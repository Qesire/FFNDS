"""Verify the compiled native HiF4 path, including non-default CUDA streams.

This is a numerical/operator check with synthetic inputs, not a model-quality
test or a speed benchmark. Both extension modules must import successfully;
the Python fallback must never count as native CUDA validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; choose a new run path")
    from HiFloat4.hif4_gpu.quant_cy import QType, quant_dequant_float
    from HiFloat4.hif4_gpu.quant_cy.base import QTensor
    from HiFloat4.hif4_gpu.quant_cy.base.cusrc import hifxg_quant, lowbit_quant

    assert QTensor.lowbit_quant is not None and QTensor.CUDA_KERNELS
    modules = {}
    for module in [hifxg_quant, lowbit_quant]:
        p = Path(module.__file__)
        modules[module.__name__] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    torch.manual_seed(20261020)
    checks = []
    for dtype in [torch.float32, torch.bfloat16]:
        source = torch.randn(257, 5120, device="cuda", dtype=dtype)
        # Include zero, signed, and large finite groups in the same check.
        source[0] = 0
        source[1] *= 100
        for desc in ["hifx4", "hifx4_1"]:
            q = QType(desc).dim(-1)
            assert QTensor.get_cuda_func(source, q) is not None
            ref = quant_dequant_float(source, q, force_fp32=True)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                staged = torch.zeros_like(source)
                # Force a producer dependency before QDQ. An accidental launch
                # on the default stream will race this copy and fail equality.
                torch.cuda._sleep(20_000_000)
                staged.copy_(source)
                actual = quant_dequant_float(staged, q, force_fp32=True)
                consumed = actual.square().sum()
            stream.synchronize()
            assert torch.equal(ref, actual), (str(dtype), desc, (ref - actual).abs().max().item())
            assert torch.isfinite(actual).all() and torch.isfinite(consumed)
            # Exercise the BF16 wrapper separately: force_fp32=False.
            native = quant_dequant_float(source, q, force_fp32=False)
            torch.cuda.synchronize()
            checks.append({"dtype": str(dtype), "format": desc, "shape": list(source.shape),
                           "stream_equal": True, "finite": True,
                           "native_vs_force_fp32_max_abs": (native.float() - ref.float()).abs().max().item()})
    output = {"status": "PASS", "scope": "synthetic native CUDA QDQ/current-stream numerical validation only",
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "modules": modules, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
