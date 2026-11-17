#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:25:00

set -euo pipefail
stage=$1
scope_sha=$2
cd "$stage"
sha256sum -c SHA256SUMS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$PWD/triton_cache"
python="$HOME/miniconda3/envs/hif4/bin/python3.11"
"$python" -B - "$scope_sha" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from runtime_guard import require_compute_allocation

digest = lambda name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
scope = json.load(open('scope.json'))
active = json.load(open('ACTIVE_RESEARCH.json'))[scope['scope_id'].lower()]
review = json.load(open('REVIEW.json'))
if not (active['runnable'] and active['scope_sha256'] == sys.argv[1] == digest('scope.json')
        and active['review_sha256'] == digest('REVIEW.json') and review['recheck_disposition'] == 'APPROVE'):
    raise RuntimeError('unreviewed module scope')
guard = require_compute_allocation()
if guard['memory_mib'] > 24 * 1024 or 'Partition=P107-A100 ' not in guard['scheduler_job_record']:
    raise RuntimeError('unexpected GPU allocation')
print(guard['scheduler_job_record'])
PY
mkdir evidence
trap 'rc=$?; printf "{\"exit_code\":%s,\"slurm_job_id\":\"%s\",\"end_utc\":\"%s\"}\n" "$rc" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > evidence/EXIT.json; sha256sum evidence/*.json evidence/*.log > evidence/SHA256SUMS' EXIT
library=$("$python" -B -c 'import json; print(json.load(open("scope.json"))["library"])')
lib_sha=$("$python" -B -c 'import json; print(json.load(open("scope.json"))["library_sha256"])')
printf '{"scope":"FFNDS_STRICT_CHIP_MODULE_BOUNDED_V1","slurm_job_id":"%s","host":"%s","start_utc":"%s","wan_nfe":0,"videos":0}\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" > evidence/RUN.json
"$python" -B chip_module_bounded_oracle_v1.py --source author/src/chipmunk --library "$library" \
    --library-sha256 "$lib_sha" --output evidence/MODULE.json > evidence/probe.log 2>&1
