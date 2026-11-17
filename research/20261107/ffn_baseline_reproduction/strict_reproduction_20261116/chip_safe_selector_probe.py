"""Validate the separately named safe Chip selector on synthetic aligned rows."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from build_chip_safe_selector import materialize
from chip_selector_probe import healthy, select, sha256


def validate_prefix(indices, counts, must_include, required_count):
    count = counts.item()
    used = set(indices.flatten()[:count].tolist())
    if count != required_count or len(used) != count or not must_include <= used:
        raise AssertionError(f"selector prefix failed: {count=} unique={len(used)}")
    if not all(0 <= i < indices.shape[-1] for i in used):
        raise AssertionError("selector returned an invalid input column")
    return count


def random_fixture():
    x = torch.zeros((1, 1, 1024), dtype=torch.bfloat16, device="cuda")
    x[..., 512:] = 1
    high = set(range(512, 1024))
    no_random = validate_prefix(*select(x, 0.0), high, 512)
    with_random = validate_prefix(*select(x, 0.05), high, 768)
    return {"without_random": no_random, "with_random": with_random}


def repaired_adversary():
    x = torch.ones((1, 1, 13824), dtype=torch.bfloat16, device="cuda")
    x[..., 1024::1024] = 0
    indices, counts = select(x, 0.0)
    count = validate_prefix(indices, counts, set(range(13824)), 13824)
    cache = torch.full_like(x, -2)
    torch.cuda.synchronize()
    torch.ops.chipmunk.copy_indices(x, cache, indices, counts)
    torch.cuda.synchronize()
    if not torch.equal(cache, x):
        raise AssertionError("repaired-prefix selective writeback mismatch")
    return {"count": count, "unique": count, "unwritten_slots": 0, "copy_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-safe-source", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("one Slurm GPU allocation required")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (8, 0) or "A100" not in props.name or torch.__version__ != "2.10.0+cu128":
        raise RuntimeError("A100 sm80 and frozen Torch CUDA runtime required")
    args.build.mkdir()
    safe_source = args.build / "chip_safe_topk.cu"
    materialize(args.author / "csrc/indexed_io/topk_indices.cu", safe_source)
    if sha256(safe_source) != args.expected_safe_source:
        raise RuntimeError("derived source differs from reviewed bytes")
    copy_source = args.author / "csrc/indexed_io/copy_indices.cu"
    binary = Path(load(
        name="ffnds_safe_selector_b5e8d1c_v1", build_directory=str(args.build), with_cuda=True,
        sources=[str(Path(__file__).with_name("chip_selector_binding.cpp")), str(safe_source), str(copy_source)],
        is_python_module=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=True,
    ))
    result = {
        "variant": "CHIPMUNK_AUTHOR_MLP_SAFE_SELECTOR_V1", "compiled_source_sha256": sha256(safe_source),
        "loaded_extension_sha256": sha256(binary), "torch": torch.__version__, "device": props.name,
        "healthy": healthy(0.0), "random_term": random_fixture(), "adversarial_wan_width": repaired_adversary(),
        "selector_calls": 4, "copy_calls": 2, "wan_nfe": 0, "videos": 0,
    }
    with args.output.open("x", encoding="utf-8") as report:
        json.dump(result, report, indent=2)
        report.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
