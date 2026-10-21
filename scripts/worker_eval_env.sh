#!/usr/bin/env bash
# Called inside the GPU-locked worker_exec.sh wrapper.
set -euo pipefail
FFNDS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$FFNDS_ROOT/.venv/bin/python" "$FFNDS_ROOT/tools/worker_identity.py" --check
unset PYTHONPATH PYTHONHOME
export PATH="$FFNDS_ROOT/.runtime/envs/vbench_official/bin:/usr/bin:/bin:$PATH"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# These four local checkpoints were staged from their published sources and hashed.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec "$@"
