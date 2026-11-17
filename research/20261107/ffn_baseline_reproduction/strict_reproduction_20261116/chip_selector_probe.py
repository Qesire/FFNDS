"""Bounded author compiled selector and copy conformance on A100."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.utils.cpp_extension import load


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def select(activation, random_keys):
    indices = torch.full_like(activation, -1, dtype=torch.int32)
    counts = torch.full(activation.shape[:2], -1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    torch.ops.chipmunk.topk_indices(activation, indices, counts, 0.7, 256, random_keys)
    torch.cuda.synchronize()
    return indices, counts


def healthy(random_keys):
    activation = torch.zeros((1, 2, 1024), dtype=torch.bfloat16, device="cuda")
    activation[0, 0, 716:] = 1
    indices, counts = select(activation, random_keys)
    masks = []
    for block, count in enumerate(counts[0].tolist()):
        valid = indices[0, block, :count].tolist()
        if count not in (512, 768, 1024) or len(set(valid)) != count or any(not 0 <= idx < 1024 for idx in valid):
            raise AssertionError(f"unsafe healthy fixture block {block}: count={count}")
        masks.append(set(valid))
    if not set(range(716, 1024)) <= masks[0] or masks[1] != set(range(1024)):
        raise AssertionError("threshold support absent")
    if random_keys == 0 and counts[0].tolist() != [512, 1024]:
        raise AssertionError("healthy selector count differs from exact threshold and padding")
    cache = torch.full_like(activation, -2)
    expected = cache.clone()
    for block, consumed in enumerate(masks):
        positions = sorted(consumed)
        expected[0, block, positions] = activation[0, block, positions]
    torch.cuda.synchronize()
    torch.ops.chipmunk.copy_indices(activation, cache, indices, counts)
    torch.cuda.synchronize()
    if not torch.equal(cache, expected):
        raise AssertionError("selective cache writeback mismatch")
    return {"random_keys": random_keys, "counts": counts[0].tolist(), "copy_exact": True}


def adversary():
    activation = torch.ones((1, 1, 13824), dtype=torch.bfloat16, device="cuda")
    activation[..., 1024::1024] = 0
    indices, counts = select(activation, 0.0)
    count = counts.item()
    values = indices.flatten().tolist()[:count]
    missing = values.count(-1)
    invalid_columns = set(range(1024, 13824, 1024))
    observed = {idx for idx in values if idx >= 0}
    expected = (set(range(13824)) - invalid_columns) | {13312}
    # Copy and MLP are intentionally absent from this diagnostic path.
    if count != 13824 or missing != 12 or observed != expected:
        raise AssertionError(f"unexpected adversary outcome: count={count}, unwritten={missing}")
    return {"reported_count": count, "unwritten_slots": missing,
            "index_of_only_padding_writer": 13312, "author_selector_disposition": "FAIL_UNSAFE_COUNT"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("one Slurm GPU allocation required")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (8, 0) or "A100" not in props.name:
        raise RuntimeError(f"expected A100/sm80, got {props.name}")
    args.build.mkdir()
    binding = Path(__file__).with_name("chip_selector_binding.cpp")
    author_units = [args.author / "csrc/indexed_io" / name for name in ("topk_indices.cu", "copy_indices.cu")]
    lib = Path(load(
        name="ffnds_author_selector_b5e8d1c", sources=[str(p) for p in [binding, *author_units]],
        build_directory=str(args.build), with_cuda=True, is_python_module=False,
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=True,
    ))
    report = {
        "scope": "author CUDA selector and copy translation units; local Torch registration",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": props.name,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).strip(),
        "host_compiler": subprocess.check_output([os.environ.get("CXX", "c++"), "--version"], text=True).splitlines()[0],
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0],
        "extension_sha256": sha256(lib), "source_sha256": {p.name: sha256(p) for p in [binding, *author_units]},
        "healthy": [healthy(0.0), healthy(0.05)], "adversary": adversary(),
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
