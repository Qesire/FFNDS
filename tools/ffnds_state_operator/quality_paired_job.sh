#!/bin/bash
# Execute only inside the exact independently approved Slurm allocation.
set -euo pipefail
stage=$1
digest=$2
cd "$stage"
[[ $(sha256sum scope.json | cut -d ' ' -f 1) == "$digest" ]]
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python_executable="${FFNDS_QUALITY_PYTHON:-$HOME/miniconda3/envs/vbench_i2v_official/bin/python3.10}"
exec "$python_executable" -B quality_paired.py run scope.json "$digest"
