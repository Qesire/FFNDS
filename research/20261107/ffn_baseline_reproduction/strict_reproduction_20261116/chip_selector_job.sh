#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
stage=$1
fingerprint=$2
cd "$stage"
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_GPUS:-} != *,* ]]
sha256sum -c SHA256SUMS
python_root="$HOME/miniconda3/envs/hif4"
export PATH="$python_root/bin:$PATH" CUDA_HOME="$python_root"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$python_root/bin/python3.11" -B - "$fingerprint" <<'PY'
import json
import sys
from importlib.metadata import version

scope = json.load(open('ACTIVE_RESEARCH.json'))['ffnds_strict_chip_selector_a100_v1']
review = json.load(open('REVIEW.json'))
if not scope['runnable'] or review['recheck_disposition'] != 'APPROVE':
    raise RuntimeError('scope is not runnable')
if scope['approval_scope_fingerprint'] != sys.argv[1] or review['approval_scope_fingerprint'] != sys.argv[1]:
    raise RuntimeError('identity mismatch')
if sys.version_info[:2] != (3, 11) or version('torch').split('+')[0] != '2.10.0':
    raise RuntimeError('Python/Torch environment mismatch')
PY
mkdir evidence
trap 'rc=$?; printf "{\"exit_code\":%s,\"slurm_job_id\":\"%s\",\"end_utc\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > evidence/EXIT.json; sha256sum evidence/*.json evidence/*.log > evidence/SHA256SUMS' EXIT
printf '{"scope":"FFNDS_STRICT_CHIP_SELECTOR_A100_V1","slurm_job_id":"%s","host":"%s","start_utc":"%s","wan_nfe":0,"videos":0}\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" > evidence/RUN.json
"$python_root/bin/python3.11" -B chip_selector_probe.py --author author --build build --output evidence/SELECTOR.json \
    > evidence/probe.log 2>&1
