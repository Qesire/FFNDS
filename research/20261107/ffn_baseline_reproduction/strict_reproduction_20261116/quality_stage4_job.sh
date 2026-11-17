#!/bin/bash
# A reviewed caller supplies the exact Slurm resources; this file never submits.
set -euo pipefail
stage=$1
scope_sha=$2
cd "$stage"
sha256sum -c SHA256SUMS
[[ $(sha256sum scope.json | cut -d' ' -f1) == "$scope_sha" ]]
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /public/home/scc/pb23061276/miniconda3/envs/vbench_i2v_official/bin/python3.10 -B quality_guard.py "$PWD/scope.json" "$scope_sha"
