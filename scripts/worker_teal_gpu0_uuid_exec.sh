#!/usr/bin/env bash
# Fixed GPU 0, same cooperative lock as worker_wait_uuid_idle_exec.sh.
# A busy device or lock fails this one attempt; no queue or retry.
set -euo pipefail
[[ "$#" -gt 0 ]] || exit 2
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$root/.venv/bin/python" "$root/tools/worker_identity.py" --check
exec 9>"/tmp/ffnds-${USER}-gpu0.lock"
flock -n 9 || exit 3
occupants="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$occupants" ]] || exit 4
uuid="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader,nounits)"
[[ "$uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || exit 5
export FFNDS_PHYSICAL_GPU=0
export CUDA_VISIBLE_DEVICES="$uuid"
unset PYTHONPATH PYTHONHOME
exec "$@"
