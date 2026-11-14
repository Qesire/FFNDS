#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@60
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
stage="$1"
scope_sha="$2"
trap 'rc=$?; printf "{\"shell_exit_code\":%s,\"job_id\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" > "$stage/SHELL_EXIT.json"' EXIT
/public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11 -B "$stage/guarded_runner_b.py" --stage "$stage" --scope-sha "$scope_sha"