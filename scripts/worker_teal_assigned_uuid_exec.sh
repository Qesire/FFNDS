#!/usr/bin/env bash
# Exactly one assigned physical index and UUID, one cooperative attempt, no retry.
set -euo pipefail
[[ "$#" -ge 3 ]] || exit 2
gpu="$1"; uuid="$2"; shift 2
[[ "$gpu" =~ ^[0-3]$ && "$uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || exit 2
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$root/.venv/bin/python" -B "$root/tools/worker_identity.py" --check
exec 9>"/tmp/ffnds-${USER}-gpu${gpu}.lock"
flock -n 9 || exit 3
actual="$(/usr/bin/nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits)"
[[ "$actual" == "$uuid" ]] || exit 5
occupants="$(/usr/bin/nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$occupants" ]] || exit 4
export FFNDS_PHYSICAL_GPU="$gpu" CUDA_VISIBLE_DEVICES="$uuid"
unset PYTHONPATH PYTHONHOME
exec "$@"
