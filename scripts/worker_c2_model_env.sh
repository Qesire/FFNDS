#!/usr/bin/env bash
# Restore the native project import path and preserve the selected CUDA UUID.
set -euo pipefail
FFNDS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$FFNDS_ROOT/.venv/bin/python" "$FFNDS_ROOT/tools/worker_identity.py" --check
task_cuda_value="${CUDA_VISIBLE_DEVICES-}"
task_cuda_was_set="${CUDA_VISIBLE_DEVICES+x}"
unset PYTHONPATH PYTHONHOME
source "$FFNDS_ROOT/config/worker.env"
if [[ "$task_cuda_was_set" == x ]]; then
    export CUDA_VISIBLE_DEVICES="$task_cuda_value"
else
    unset CUDA_VISIBLE_DEVICES
fi
unset SKIP_PLATFORM_CHECK
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$FFNDS_ROOT"
exec "$@"
