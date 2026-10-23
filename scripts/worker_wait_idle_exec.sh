#!/usr/bin/env bash
# Wait without importing a model; retain the GPU lock through the single exec.
set -euo pipefail
task_status="${1:?Expected private queue status path}"
shift
[[ "$#" -gt 0 ]] || { echo 'Expected one child command.' >&2; exit 2; }
FFNDS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$FFNDS_ROOT/.venv/bin/python" "$FFNDS_ROOT/tools/worker_identity.py" --check
export FFNDS_ROOT
unset PYTHONPATH PYTHONHOME
source "$FFNDS_ROOT/config/worker.env"
export CUDA_VISIBLE_DEVICES=''
cd "$FFNDS_ROOT"

queue_receipt() {
    "$FFNDS_ROOT/.venv/bin/python" -c '
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix(".tmp")
r={"status":sys.argv[2],"selected_GPU":None if sys.argv[3]=="" else int(sys.argv[3]),
   "heartbeat_epoch_seconds":time.time(),"automatic_model_retry":False}
t.write_text(json.dumps(r,indent=2)+"\n");t.replace(p)
' "$task_status" "$1" "${2:-}"
}

while true; do
    task_indices="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)"
    [[ -n "$task_indices" ]] || { echo 'No GPU inventory; refusing to launch.' >&2; exit 3; }
    while read -r task_gpu; do
        [[ "$task_gpu" =~ ^[0-3]$ ]] || { echo 'Unexpected physical GPU index.' >&2; exit 3; }
        exec 9>"/tmp/ffnds-${USER}-gpu${task_gpu}.lock"
        if ! flock -n 9; then exec 9>&-; continue; fi
        task_occupants="$(nvidia-smi -i "$task_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
        if [[ -n "$task_occupants" ]]; then exec 9>&-; continue; fi
        export CUDA_VISIBLE_DEVICES="$task_gpu"
        queue_receipt 'IDLE_GPU_LOCKED_STARTING_SINGLE_ATTEMPT' "$task_gpu"
        echo 'One idle GPU locked; starting the single authorized attempt.'
        exec "$@"
    done <<< "$task_indices"
    queue_receipt 'WAITING_FOR_IDLE_GPU_NO_MODEL_STARTED'
    sleep 15
done
