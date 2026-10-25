"""Compare the original CPU HiF4 reference with accepted native paired values.

No CUDA package import or model execution. Equality here is scoped to captured
operands and decoded BF16 outputs, not every internal GPU scale register.
"""
import argparse
import json
from pathlib import Path

from analyze_ffnds_observations import load_arrays
from audit_hif4_group_separability import QROOT, ROOT, reference
from ffnds_observation_common import sha256, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve first reference comparison")
    run = json.loads(args.run.read_text())
    if run["status"] != "PASS_PASSIVE_CAPTURE_NOOP":
        raise ValueError("Require accepted native observations")
    import torch
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only reference check")
    qtype, quantize = reference()
    result = {"status": "COMPLETE_CAPTURED_NATIVE_VS_ORIGINAL_CPU_REFERENCE",
              "source_id": run["source_id"], "run_sha256": sha256(args.run),
              "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
                  Path(__file__), ROOT / "tools/audit_hif4_group_separability.py",
                  ROOT / QROOT / "QType.py", ROOT / QROOT / "QFuncs/hifx.py",
                  ROOT / QROOT / "cusrc/hifxg_quant_cuda.cu"]},
              "cases": [], "GPU_calls": 0, "model_calls": 0,
              "internal_GPU_metadata_exported": False}
    for scope in run["scopes"]:
        path = args.run.parent / "observed" / scope["path"]
        if sha256(path) != scope["sha256"]:
            raise ValueError("Native payload changed")
        arrays = load_arrays(path)
        for stage in ("q1", "q2"):
            pre = torch.from_numpy(arrays[stage + "_pre"]).bfloat16()
            actual = torch.from_numpy(arrays[stage + "_post"]).bfloat16()
            expected = quantize(pre.float(), qtype, -1).bfloat16()
            mismatch = expected.view(torch.int16) != actual.view(torch.int16)
            zero_only = mismatch & (expected == 0) & (actual == 0)
            sign_preserved = torch.copysign(expected, pre)
            fixed_mismatch = sign_preserved.view(torch.int16) != actual.view(torch.int16)
            result["cases"].append({"step": scope["step"], "block": scope["block"],
                "stage": stage, "shape": list(actual.shape),
                "BF16_bit_mismatches": int(mismatch.sum()),
                "signed_zero_only_mismatches": int(zero_only.sum()),
                "mismatches_with_negative_zero_input": int((mismatch & (pre == 0) & torch.signbit(pre)).sum()),
                "BF16_bit_mismatches_after_preserving_input_sign": int(fixed_mismatch.sum()),
                "nonzero_value_mismatches": int((mismatch & ~zero_only).sum()),
                "max_abs_difference": float((expected.float() - actual.float()).abs().max())})
    result["CPU_reference_calls"] = len(result["cases"])
    result["all_captured_BF16_bits_equal"] = all(c["BF16_bit_mismatches"] == 0 for c in result["cases"])
    result["all_captured_values_equal"] = all(c["nonzero_value_mismatches"] == 0 for c in result["cases"])
    result["all_bits_equal_after_preserving_input_sign"] = all(c["BF16_bit_mismatches_after_preserving_input_sign"] == 0 for c in result["cases"])
    result["CUDA_initialized"] = torch.cuda.is_initialized()
    write_json(args.out, result)


if __name__ == "__main__":
    main()
