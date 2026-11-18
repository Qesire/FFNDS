#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60

set -euo pipefail
stage=$1
scope_sha=$2
cd "$stage"
sha256sum -c SHA256SUMS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$PWD/triton_cache"
mkdir receipt
trap 'rc=$?; printf "{\"exit_code\":%s,\"slurm_job_id\":\"%s\",\"end_utc\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > receipt/EXIT.json; sha256sum receipt/*.json receipt/*.log > receipt/SHA256SUMS' EXIT
worker=
terminate() {
    local code=$1
    if [[ -n "$worker" ]]; then
        kill -TERM "$worker" 2>/dev/null || true
        wait "$worker" || true
    fi
    exit "$code"
}
trap 'terminate 143' TERM
trap 'terminate 130' INT
"$HOME/miniconda3/envs/hif4/bin/python3.11" -B wan_state_operator_job.py "$scope_sha" > receipt/runner.log 2>&1 &
worker=$!
wait "$worker"
