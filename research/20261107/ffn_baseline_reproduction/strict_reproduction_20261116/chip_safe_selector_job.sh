#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
stage=$1
safe_source_sha=$2
cd "$stage"
sha256sum -c SHA256SUMS
python_root="$HOME/miniconda3/envs/hif4"
export PATH="$python_root/bin:$PATH" CUDA_HOME="$python_root"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$python_root/bin/python3.11" -B - "$safe_source_sha" <<'PY'
import hashlib
import json
import sys

from runtime_guard import require_compute_allocation

scope = json.load(open('ACTIVE_RESEARCH.json'))['ffnds_strict_chip_safe_selector_a100_v1']
review_bytes = open('REVIEW.json', 'rb').read()
review = json.loads(review_bytes)
if not scope['runnable'] or review['recheck_disposition'] != 'APPROVE':
    raise RuntimeError('scope is not runnable')
if scope['review_sha256'] != hashlib.sha256(review_bytes).hexdigest() or review['safe_cuda_sha256'] != sys.argv[1]:
    raise RuntimeError('identity mismatch')
allocation = require_compute_allocation()
if allocation['memory_mib'] > 24 * 1024:
    raise RuntimeError('unexpected memory allocation')
PY
mkdir evidence
trap 'rc=$?; printf "{\"exit_code\":%s,\"slurm_job_id\":\"%s\",\"end_utc\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > evidence/EXIT.json; sha256sum evidence/*.json evidence/*.log > evidence/SHA256SUMS' EXIT
printf '{"scope":"FFNDS_STRICT_CHIP_SAFE_SELECTOR_A100_V1","slurm_job_id":"%s","host":"%s","start_utc":"%s","wan_nfe":0,"videos":0}\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" > evidence/RUN.json
"$python_root/bin/python3.11" -B chip_safe_selector_probe.py --author author --build build \
    --output evidence/SAFE_SELECTOR.json --expected-safe-source "$safe_source_sha" > evidence/probe.log 2>&1
