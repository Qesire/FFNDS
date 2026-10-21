#!/usr/bin/env bash
# Called through run_on_worker.py. The home directory is shared between hosts.
set -euo pipefail
gpu="${1:?Expected physical GPU index 0 or 1}"
shift
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo 'Only GPUs 0/1 are allowed.' >&2; exit 2; }
FFNDS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$FFNDS_ROOT/.venv/bin/python" "$FFNDS_ROOT/tools/worker_identity.py" --check
export FFNDS_ROOT
unset PYTHONPATH PYTHONHOME
source "$FFNDS_ROOT/config/worker.env"
export CUDA_VISIBLE_DEVICES="$gpu"
exec 9>"/tmp/ffnds-${USER}-gpu${gpu}.lock"
flock -n 9 || { echo "Another FFNDS run holds GPU $gpu." >&2; exit 3; }
occupants="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "$occupants" ]]; then
    echo "GPU $gpu already has compute processes: $occupants" >&2
    exit 3
fi
cd "$FFNDS_ROOT"
exec "$@"
