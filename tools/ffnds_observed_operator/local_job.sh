#!/usr/bin/env bash
set -euo pipefail

stage=$1
scope_sha=$2
cd "$stage"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TRITON_CACHE_DIR="$stage/triton_cache"
export TOKENIZERS_PARALLELISM=false
python_bin=/public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11
child=
finish() {
    local rc=$?
    printf '{"slurm_job_id":"%s","scope_sha256":"%s","exit_code":%s,"end_utc":"%s"}\n' \
        "$SLURM_JOB_ID" "$scope_sha" "$rc" "$(date -u +%FT%TZ)" > ATTEMPT_EXIT.json
}
terminate() {
    if [[ -n "$child" ]]; then
        # The guard and its worker share the process group started by setsid.
        kill -TERM -- "-$child" 2>/dev/null || true
        wait "$child" || true
    fi
    exit 143
}
trap finish EXIT
trap terminate TERM INT
sha256sum -c SHA256SUMS
setsid "$python_bin" -B scope_job.py "$stage" "$scope_sha" > worker.log 2>&1 &
child=$!
wait "$child"
