#!/usr/bin/env bash
# The current executor can have four physical GPUs. Keep the historical
# two-GPU launcher unchanged so its registered source bindings remain valid.
set -euo pipefail
gpu="${1:?Expected physical GPU index 0, 1, 2 or 3}"
shift
[[ "$gpu" =~ ^[0-3]$ ]] || { echo 'Expected one physical GPU index 0..3.' >&2; exit 2; }
FFNDS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$FFNDS_ROOT/.venv/bin/python" "$FFNDS_ROOT/tools/worker_identity.py" --check
export FFNDS_ROOT
unset PYTHONPATH PYTHONHOME
source "$FFNDS_ROOT/config/worker.env"
exec 9>"/tmp/ffnds-${USER}-gpu${gpu}.lock"
flock -n 9 || { echo "Another FFNDS run holds GPU $gpu." >&2; exit 3; }
# A missing device, query failure, or any existing compute process fails closed.
occupants="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "$occupants" ]]; then
    echo "GPU $gpu already has compute processes: $occupants" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$gpu"
cd "$FFNDS_ROOT"
exec "$@"
