"""Derive the Wan-width safe selector from the byte-pinned author CUDA unit."""

import argparse
import hashlib
from pathlib import Path


AUTHOR_SHA256 = "05407a642dcb5f201b605547d53b91effb15bb9b38fce3344207c8f4204ae251"


def materialize(source, destination):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUTHOR_SHA256:
        raise ValueError("the author selector source does not match the pinned commit")
    code = raw.decode("utf-8").replace("\r\n", "\n")
    edits = (
        ("#include <torch/extension.h>", "#include <torch/extension.h>\n#include <c10/cuda/CUDAStream.h>"),
        ("int last_invalid_value = -1;", "int last_invalid_value = -1;\n    unsigned int rejected_lanes = 0;"),
        ("last_invalid_value = col;", "last_invalid_value = col;\n            rejected_lanes |= 1u << (col / NUM_THREADS);"),
        ("        indices[pos] = last_invalid_value;\n    }", """        if (pos < base_count_shared + remainder_shared) {
            indices[pos] = last_invalid_value;
        }
    }
    __syncthreads();

    // Each thread first contributes the same last rejected column as upstream.
    // If too few threads had rejected columns, use earlier rejects as padding.
    // All last rejects were written in that case, so exclude each of them.
    const unsigned int target = base_count_shared + remainder_shared;
    const unsigned int count_after_last = count;
    __syncthreads();
    if (count_after_last < target) {
        for (int col = tid; col < cols; col += NUM_THREADS) {
            if (col != last_invalid_value && (rejected_lanes & (1u << (col / NUM_THREADS)))) {
                unsigned int pos = atomicAdd(&count, 1);
                if (pos < target) {
                    indices[pos] = col;
                }
            }
        }
    }"""),
        ("    const auto cols  = activation.size(2);", """    const auto cols  = activation.size(2);
    TORCH_CHECK(activation.is_cuda() && indices.is_cuda() && counts.is_cuda(), "expected CUDA inputs");
    TORCH_CHECK(activation.device() == indices.device() && activation.device() == counts.device() &&
                activation.get_device() == c10::cuda::current_device(), "CUDA devices must match the active device");
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream(activation.get_device()) ==
                c10::cuda::getDefaultCUDAStream(activation.get_device()), "the author launcher uses the default stream");
    TORCH_CHECK(activation.is_contiguous() && indices.is_contiguous() && counts.is_contiguous(),
                "expected contiguous inputs");
    TORCH_CHECK(indices.sizes() == activation.sizes() && counts.size(0) == batch && counts.size(1) == rows,
                "output shape must match the activation rows or columns");
    TORCH_CHECK(batch > 0 && rows > 0 && cols >= SAMPLE_SIZE && cols <= 32 * NUM_THREADS &&
                multiple_of == 256 && cols % multiple_of == 0, "unsupported selector geometry or alignment");
    TORCH_CHECK(random_amount >= 0 && random_amount <= 1, "invalid random probability");"""),
    )
    for old, new in edits:
        if code.count(old) != 1:
            raise RuntimeError("author selector edit did not bind to a unique source location")
        code = code.replace(old, new, 1)
    code = "// FFNDS safe selector variant derived from sandyresearch/chipmunk b5e8d1c.\n" + code
    with destination.open("xb") as out:
        out.write(code.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    materialize(args.original, args.destination)
