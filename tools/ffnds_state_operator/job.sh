#!/bin/bash
set -euo pipefail
stage="$1"
scope_sha="$2"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
cd "$stage"
trap 'rc=$?; printf "{\"exit_code\":%s,\"slurm_job_id\":\"%s\",\"end_utc\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "shell-$SLURM_JOB_ID.json"' EXIT
"/public/home/scc/$USER/miniconda3/envs/hif4/bin/python3.11" -B scope_job.py "$stage" "$scope_sha"

